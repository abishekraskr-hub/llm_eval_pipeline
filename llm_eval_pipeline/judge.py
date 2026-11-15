"""The judge: a second LLM that grades the app under test.

DeepEval defaults to OpenAI, so a custom DeepEvalBaseLLM is the hook that lets it
grade with another provider. Four things matter here beyond plumbing:

1. **Provider is config, not code.** Groq and Gemini both expose an
   OpenAI-compatible endpoint, so switching is a base_url and a key -- see
   PROVIDERS. The judge must never be the reason you cannot change provider.

2. **DeepEval's metrics want *structured* output** (a pydantic schema). Providers
   disagree about how to give it: strict `json_schema` is supported by Gemini and
   by only some Groq models, while `json_object` mode is near-universal. We try
   the strict path once, remember whether it worked, and fall back to JSON mode
   plus manual validation. Without the fallback, judging on
   `llama-3.3-70b-versatile` fails on every single metric call.

3. **Judge on a different provider from the app under test.** The app runs on
   Gemini, so judging on Groq means the two never share a quota bucket -- a full
   100-question run stops being a scheduling problem. It also avoids a model
   grading its own output, which is a known bias.

4. **Not every 429 is fatal.** A per-DAY limit cannot be waited out in-process and
   must abort the run. A per-MINUTE limit (Groq's TPM ceiling is the one you will
   actually hit) clears in seconds and must be retried, or a healthy run dies for
   no reason. These are told apart by the provider's own error text.
"""
import json
import os
import re
import threading
import time
from collections import deque

from deepeval.models.base_model import DeepEvalBaseLLM
from openai import APIStatusError, BadRequestError, OpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

# Both speak the OpenAI wire format, so the only per-provider facts are these.
PROVIDERS: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        # One of the few Groq models with STRICT json_schema support, which is
        # what DeepEval needs. See eval_config.yaml for the alternatives.
        "default_model": "openai/gpt-oss-120b",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash-lite",
    },
}

# Phrases that mean "this limit resets tomorrow". Everything else in a 429 is
# treated as a per-minute ceiling and retried.
_FATAL_LIMIT_MARKERS = ("per day", "requests per day", "rpd", "tpd", "daily")

# Groq includes the wait in the error text: "Please try again in 3.456s".
_RETRY_AFTER = re.compile(r"try again in ([\d.]+)\s*s", re.IGNORECASE)

_JSON_MODE_INSTRUCTION = (
    "\n\nReturn ONLY a single valid JSON object and nothing else -- no markdown "
    "fences, no explanation, no preamble. It must conform to this JSON schema:\n"
    "{schema}\n"
)

# A judge that wrapped its JSON in prose or fences is salvageable; pull the
# outermost object out rather than failing the metric over formatting.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class JudgeQuotaExceeded(RuntimeError):
    """The judge is out of quota for the day. Never reported as a quality failure."""


class _RateLimiter:
    """Thread-safe rolling-window limiter. DeepEval fans metrics out across threads,
    so the window has to be shared rather than per-caller."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._stamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._stamps and now - self._stamps[0] >= 60.0:
                    self._stamps.popleft()
                if len(self._stamps) < self.per_minute:
                    self._stamps.append(now)
                    return
                wait = 60.0 - (now - self._stamps[0]) + 0.5
            time.sleep(max(wait, 0.5))


def _is_daily_limit(message: str) -> bool:
    low = message.lower()
    return any(marker in low for marker in _FATAL_LIMIT_MARKERS)


def _retry_after_seconds(exc: RateLimitError, default: float = 20.0) -> float:
    """How long to wait out a per-minute 429.

    Groq sets a `retry-after` header on every 429 and also states the wait in the
    error text. Prefer the header, fall back to the text, then to a fixed wait.
    A second of headroom, because the window is measured on the server's clock.
    """
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            header = response.headers.get("retry-after")
        except Exception:
            header = None
        if header:
            try:
                return min(float(header) + 1.0, 65.0)
            except (TypeError, ValueError):
                pass
    match = _RETRY_AFTER.search(str(exc))
    if match:
        try:
            return min(float(match.group(1)) + 1.0, 65.0)
        except ValueError:
            pass
    return default


class LLMJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str | None = None,
        provider: str = "groq",
        api_key_env: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        requests_per_minute: int = 12,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        rate_limit_retries: int = 6,
    ):
        if provider not in PROVIDERS:
            raise RuntimeError(
                f"unknown judge provider {provider!r}; known: {sorted(PROVIDERS)}"
            )
        preset = PROVIDERS[provider]

        self.provider = provider
        self.model_name = model or preset["default_model"]
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.rate_limit_retries = rate_limit_retries
        self.limiter = _RateLimiter(requests_per_minute)

        key_name = api_key_env or preset["api_key_env"]
        key = os.getenv(key_name)
        if not key:
            raise RuntimeError(
                f"{key_name} is not set; the judge needs it. Put it in .env "
                f"(see .env.example) or set it as a CI secret."
            )
        self.api_key_env = key_name

        self._client = OpenAI(
            base_url=base_url or preset["base_url"],
            api_key=key,
            # Transient blips only. Rate limits and day-quota are handled below,
            # where we can tell the two apart.
            max_retries=max_retries,
            timeout=timeout_s,
        )
        # Strict json_schema support is per-model AND per-schema: strict mode
        # requires every field to be required, which some DeepEval schemas break.
        # Remember which schemas were rejected, so each is probed at most once.
        self._no_strict: set[str] = set()
        # Counted so a run can report what grading actually cost in requests.
        self.calls = 0
        super().__init__(self.model_name)

    # -- DeepEvalBaseLLM interface -------------------------------------------------
    def load_model(self):
        return self._client

    def get_model_name(self) -> str:
        return f"{self.provider}:{self.model_name}"

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs):
        if schema is None:
            return self._call(lambda: self._plain(prompt))
        if schema.__name__ in self._no_strict:
            return self._call(lambda: self._json_mode(prompt, schema))
        try:
            return self._call(lambda: self._strict_schema(prompt, schema))
        except _SchemaUnsupported:
            # Remembered per schema, so every later call with it skips the probe.
            self._no_strict.add(schema.__name__)
            return self._call(lambda: self._json_mode(prompt, schema))

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs):
        # The sync client under a rolling rate limit is what we want here; real
        # concurrency would just burn the per-minute quota faster than it refills.
        return self.generate(prompt, schema=schema, **kwargs)

    # -- request plumbing ----------------------------------------------------------
    def _call(self, fn):
        """Rate-limit, count, and retry a single judge request.

        A per-minute 429 is slept off and retried. A per-day 429 aborts the run.
        """
        attempt = 0
        while True:
            self.limiter.acquire()
            self.calls += 1
            try:
                return fn()
            except RateLimitError as exc:
                message = str(exc)
                if _is_daily_limit(message):
                    raise JudgeQuotaExceeded(
                        f"judge {self.get_model_name()} hit a per-day limit. "
                        f"Try another model or enable billing. {message}"
                    ) from exc
                attempt += 1
                if attempt > self.rate_limit_retries:
                    raise JudgeQuotaExceeded(
                        f"judge {self.get_model_name()} kept hitting a per-minute "
                        f"limit after {self.rate_limit_retries} retries. Lower "
                        f"judge.requests_per_minute in eval_config.yaml. {message}"
                    ) from exc
                time.sleep(_retry_after_seconds(exc))

    def _strict_schema(self, prompt: str, schema: type[BaseModel]):
        """Provider-native structured output. Not all Groq models support it."""
        parse = getattr(self._client.chat.completions, "parse", None)
        if parse is None:  # older SDKs keep it under .beta
            parse = self._client.beta.chat.completions.parse
        try:
            resp = parse(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=schema,
                temperature=self.temperature,
            )
        except (BadRequestError, NotImplementedError) as exc:
            raise _SchemaUnsupported(str(exc)) from exc
        except APIStatusError as exc:
            # 400/404/422 here means "this model will not do json_schema", which is
            # a capability answer, not an outage. Anything else is a real error.
            if exc.status_code in (400, 404, 422):
                raise _SchemaUnsupported(str(exc)) from exc
            raise
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise _SchemaUnsupported("provider accepted the schema but returned nothing parsable")
        return parsed

    def _json_mode(self, prompt: str, schema: type[BaseModel]):
        """Fallback: ask for a JSON object and validate it ourselves."""
        instruction = _JSON_MODE_INSTRUCTION.format(
            schema=json.dumps(schema.model_json_schema(), indent=2)
        )
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt + instruction}],
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )
        content = (resp.choices[0].message.content or "").strip()
        return _validate(schema, content)

    def _plain(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return (resp.choices[0].message.content or "").strip()


class _SchemaUnsupported(RuntimeError):
    """This model will not honour strict json_schema; use JSON mode instead."""


def _validate(schema: type[BaseModel], content: str):
    """Validate a JSON-mode response, repairing fences and stray prose first."""
    if not content:
        raise RuntimeError("judge returned an empty response")
    try:
        return schema.model_validate_json(content)
    except ValidationError:
        pass
    match = _JSON_OBJECT.search(content)
    if match:
        try:
            return schema.model_validate_json(match.group(0))
        except ValidationError as exc:
            raise RuntimeError(
                f"judge returned JSON that does not match {schema.__name__}: {exc}"
            ) from exc
    raise RuntimeError(f"judge returned no JSON object: {content[:200]!r}")
