"""Judge registry: pick the grader by name from config.

    judge:
      provider: groq          # groq | gemini | fake
      model: openai/gpt-oss-120b
      api_key_env: GROQ_API_KEY

Adding a provider is a new file plus one line in `REGISTRY`. Nothing else in the
library names a provider.

`groq` is the default, and deliberately so: the judge must not share a quota
bucket with the app it is grading. The app under test here runs on Gemini, so
judging on Groq means a full run is never a scheduling problem between the two,
and no model ever grades its own output -- a known source of self-preference
bias.

`fake` makes no network calls at all. It exists so CI can exercise the whole
pipeline -- config, dataset, target, gate, storage, report -- on every push
without a key and without spending a cent. A library whose tests needed a live
judge would be run once and then never again.
"""
from .fake import FakeJudge, FakeScorer


class UnknownJudge(ValueError):
    """The configured provider does not exist. A config error: exit 2."""


def _build_llm(cfg: dict):
    """The real judge: an LLM behind an OpenAI-compatible endpoint."""
    from ..judge import LLMJudge
    from ..metrics import Scorer

    judge = LLMJudge(
        provider=cfg.get("provider", "groq"),
        # None lets the provider's own default model and key name apply.
        model=cfg.get("model"),
        api_key_env=cfg.get("api_key_env"),
        requests_per_minute=cfg.get("requests_per_minute", 12),
        temperature=cfg.get("temperature", 0.0),
    )
    metrics = cfg.get("metrics") or {}
    scorer = Scorer(
        judge,
        measure_faithfulness=metrics.get("faithfulness", True),
        measure_relevancy=metrics.get("answer_relevancy", True),
    )
    return judge, scorer


def _build_fake(cfg: dict):
    judge = FakeJudge(seed=str(cfg.get("seed") or "llmeval"))
    return judge, FakeScorer(judge)


# provider name -> builder returning (judge, scorer)
REGISTRY = {
    "groq": _build_llm,
    "gemini": _build_llm,
    "fake": _build_fake,
}


def build(cfg: dict | None):
    """Return `(judge, scorer)` for the configured provider."""
    cfg = dict(cfg or {})
    provider = str(cfg.get("provider") or "groq").strip().lower()
    builder = REGISTRY.get(provider)
    if builder is None:
        raise UnknownJudge(
            f"unknown judge.provider {provider!r}; available: {', '.join(sorted(REGISTRY))}"
        )
    return builder(cfg)


__all__ = ["REGISTRY", "UnknownJudge", "build", "FakeJudge", "FakeScorer"]
