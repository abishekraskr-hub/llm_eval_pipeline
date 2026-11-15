"""Evaluate any app reachable over HTTP, in any language.

This is the path that makes the library genuinely language-agnostic: the app only
has to answer one HTTP request. Nothing about it is assumed -- not the field
names, not the shape of the JSON, not even that it is running yet.

    target:
      type: http
      url: http://localhost:8000/chat
      headers: { Authorization: "Bearer ${MYAPP_TOKEN}" }
      request:
        question_field: query
      response:
        answer: reply
        retrieved: sources
        source_id: url
        content: text
      start:
        command: "python app_server.py"
        health_url: http://localhost:8000/health

Field names are MAPPED rather than mandated, because an app that already serves
users should not have to change its API to be evaluated. The `response` block
holds dot-paths into the reply, so a nested `{"data": {"answer": ...}}` works.

Not supported: streaming responses. The evaluator needs the whole answer and its
retrieved passages before it can score anything, so a token stream would have to
be fully buffered first -- which is what the non-streaming endpoint already is.
Point this at that one.
"""
import os
import re
import subprocess
import time
from typing import Any

import httpx2 as httpx

from .base import TargetResponse, TargetUnavailable

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Shapes that look like a real credential rather than a reference to one. A
# literal key in a config file ends up in git history, so this is refused
# outright rather than warned about.
_LOOKS_LIKE_SECRET = (
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),       # Google
    re.compile(r"sk-[A-Za-z0-9]{20,}"),            # OpenAI-style
    re.compile(r"gsk_[A-Za-z0-9]{30,}"),           # Groq
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),           # GitHub
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),   # Slack
)


class TargetConfigError(ValueError):
    """The http target config is wrong. Exit 2, not a run failure."""


def expand_env(value: str, where: str) -> str:
    """Replace `${VAR}` with the environment's value, and refuse literal secrets.

    A missing variable is an error rather than an empty string: silently sending
    `Bearer ` would surface as a confusing 401 from the app instead of as the
    config problem it actually is.
    """
    for pattern in _LOOKS_LIKE_SECRET:
        if pattern.search(value):
            raise TargetConfigError(
                f"{where} looks like a literal API key. Put it in an environment "
                f'variable and reference it: "${{MY_TOKEN}}". A key written into a '
                f"config file ends up in git history."
            )

    missing: list[str] = []

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in os.environ:
            missing.append(name)
            return ""
        return os.environ[name]

    out = _ENV_REF.sub(replace, value)
    if missing:
        raise TargetConfigError(
            f"{where} references ${{{missing[0]}}}, which is not set in the environment"
        )
    return out


def dig(payload: Any, path: str | None) -> Any:
    """Follow a dot-path into a nested reply. An empty path returns None."""
    if not path:
        return None
    current = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


class HttpTarget:
    def __init__(self, cfg: dict, config_dir=None):
        self.url = cfg.get("url") or cfg.get("base_url")
        if not self.url:
            raise TargetConfigError("target.url is required for type: http")
        self.method = str(cfg.get("method") or "POST").upper()
        self.timeout_s = float(cfg.get("timeout_s") or 120)
        self.retries = int(cfg.get("retries") or 0)
        self.config_dir = config_dir

        self.headers = {
            key: expand_env(str(value), f"target.headers.{key}")
            for key, value in (cfg.get("headers") or {}).items()
        }

        request = cfg.get("request") or {}
        self.question_field = str(request.get("question_field") or "question")
        self.extra_body = dict(request.get("extra_body") or {})

        # Defaults are the library's own field names, so an app that already
        # speaks them needs no `response` block at all.
        response = cfg.get("response") or {}
        self.map = {
            "answer": response.get("answer") or "answer",
            "retrieved": response.get("retrieved") or "retrieved",
            "source_id": response.get("source_id") or "source_id",
            "content": response.get("content") or "content",
            "page": response.get("page") or "page",
            "refused": response.get("refused"),
            "contexts": response.get("contexts") or "contexts",
        }
        self.start_cfg = cfg.get("start") or {}
        self._process: subprocess.Popen | None = None

    # -- optional process management ---------------------------------------

    def start(self) -> None:
        """Launch the app and wait for its health check, if configured.

        Exists so `llmeval run` is one command in CI rather than a shell script
        that starts a server, sleeps an arbitrary number of seconds and hopes.
        Polling the health URL replaces the sleep.
        """
        command = self.start_cfg.get("command")
        if not command:
            return
        health_url = self.start_cfg.get("health_url")
        timeout = float(self.start_cfg.get("startup_timeout_s") or 60)

        self._process = subprocess.Popen(
            command,
            shell=True,
            cwd=str(self.config_dir) if self.config_dir else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if not health_url:
            return

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                output = (self._process.stdout.read() if self._process.stdout else "") or ""
                raise TargetUnavailable(
                    f"target.start.command exited with code {self._process.returncode} "
                    f"before becoming healthy:\n{output[-1500:]}"
                )
            try:
                if httpx.get(health_url, timeout=5).status_code < 400:
                    return
            except Exception:
                pass
            time.sleep(0.5)
        self.stop()
        raise TargetUnavailable(
            f"target did not pass {health_url} within {timeout:g}s of starting"
        )

    def stop(self) -> None:
        """Kill the app we started. Must run even when the run failed."""
        if self._process is None:
            return
        process, self._process = self._process, None
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()

    def __enter__(self) -> "HttpTarget":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # -- the contract ------------------------------------------------------

    def ask(self, question: str) -> TargetResponse:
        body = {self.question_field: question, **self.extra_body}
        last: Exception | None = None

        # Retries cover a cold start or a transient 5xx. A 4xx is NOT retried: it
        # will fail identically next time and would only burn the budget.
        for attempt in range(self.retries + 1):
            try:
                resp = httpx.request(
                    self.method, self.url, json=body,
                    timeout=self.timeout_s, headers=self.headers,
                )
            except Exception as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise TargetUnavailable(
                    f"{self.url} unreachable: {type(exc).__name__}: {exc}"
                ) from exc

            if resp.status_code == 429:
                raise TargetUnavailable(f"target out of quota (429): {resp.text[:300]}")
            if resp.status_code >= 500 and attempt < self.retries:
                last = RuntimeError(f"HTTP {resp.status_code}")
                time.sleep(1.0 * (attempt + 1))
                continue
            if resp.status_code >= 400:
                raise TargetUnavailable(
                    f"target error [{resp.status_code}]: {resp.text[:300]}"
                )
            return self._to_response(resp)

        raise TargetUnavailable(
            f"{self.url} failed after {self.retries + 1} attempt(s): {last}"
        )

    def _to_response(self, resp) -> TargetResponse:
        try:
            payload = resp.json()
        except Exception as exc:
            raise TargetUnavailable(
                f"{self.url} did not return JSON: {resp.text[:200]}"
            ) from exc
        if not isinstance(payload, dict):
            raise TargetUnavailable(
                f"{self.url} returned {type(payload).__name__}, expected a JSON object"
            )

        answer = dig(payload, self.map["answer"])
        raw_hits = dig(payload, self.map["retrieved"])

        retrieved = None
        if isinstance(raw_hits, list):
            # Remapped onto the library's own names before `from_dict` ranks and
            # de-duplicates them, so every target goes through one code path.
            retrieved = [
                {
                    "source_id": dig(hit, self.map["source_id"]) if isinstance(hit, dict) else hit,
                    "page": dig(hit, self.map["page"]) if isinstance(hit, dict) else None,
                    "content": dig(hit, self.map["content"]) if isinstance(hit, dict) else None,
                }
                for hit in raw_hits
            ]

        contexts = dig(payload, self.map["contexts"])
        refused = dig(payload, self.map["refused"]) if self.map["refused"] else None

        latency = payload.get("latency_s")
        if latency is None and hasattr(resp, "elapsed"):
            try:
                latency = resp.elapsed.total_seconds()
            except Exception:
                latency = None

        return TargetResponse.from_dict(
            {
                "answer": answer or "",
                "contexts": contexts if isinstance(contexts, list) else [],
                "retrieved": retrieved,
                "refused": refused,
                "usage": payload.get("usage"),
                "trace": payload.get("trace"),
                "latency_s": latency or 0.0,
                "cost_usd": payload.get("cost_usd"),
                "model": payload.get("model"),
            }
        )

    def describe(self) -> dict:
        info = {"type": "http", "url": self.url, "method": self.method}
        # An optional /config endpoint lets an app declare its own model and
        # capabilities. Absence is normal, not an error.
        base = self.url.rsplit("/", 1)[0]
        for candidate in (f"{base}/config", f"{self.url.rstrip('/')}/config"):
            try:
                resp = httpx.get(candidate, timeout=5, headers=self.headers)
                if resp.status_code == 200 and isinstance(resp.json(), dict):
                    info.update(resp.json())
                    break
            except Exception:
                continue
        return info
