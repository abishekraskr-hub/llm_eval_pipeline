"""A judge that makes no network calls, for tests and for CI.

**Faithfulness is real, if crude.** It is the fraction of the answer's sentences
that actually appear in the retrieved passages. That makes it deterministic AND
meaningful: an app that quotes its sources scores 1.0, and an app that invents a
sentence is marked down for exactly that sentence. A purely random number would
have made `examples/echo_target` prove only that the pipeline runs, rather than
that the gate can tell grounded from ungrounded.

**Answer relevance is synthetic** -- a hash of (question, answer) in a high band.
There is no honest way to judge relevance without a model, so this one is only
deterministic, not meaningful, and must not be read as a measurement.

Two protections against that mistake, both load-bearing:

1. the model name is `fake:no-model`, which appears in every report;
2. `judge.provider: fake` feeds the comparability hash, so a fake run and a real
   run can never be each other's baseline. The paired test reports "no comparable
   baseline", which is the truth.
"""
import hashlib
import re

from ..metrics import ItemScores


def _unit(*parts: str) -> float:
    """A stable number in [0, 1) from the given strings."""
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _grounding(answer: str, contexts: list[str]) -> tuple[int, int]:
    """(supported, total) sentences of `answer` found in `contexts`.

    Crude on purpose -- verbatim containment, not entailment. A real judge reads
    for meaning; this only has to be deterministic and to move in the right
    direction when an app starts inventing things.
    """
    haystack = _normalise(" ".join(contexts))
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences:
        return 0, 0
    supported = sum(1 for s in sentences if _normalise(s).rstrip(".!?") in haystack)
    return supported, len(sentences)


class FakeJudge:
    """Satisfies the two things the runner asks of a judge, and calls nothing."""

    def __init__(self, seed: str = "llmeval"):
        self.seed = seed
        self.calls = 0

    def get_model_name(self) -> str:
        return "fake:no-model"


class FakeScorer:
    """Deterministic pseudo-scores. Never a measurement of anything."""

    # Relevance lands in [0.85, 1.00): above the example floors, so a healthy
    # run passes, while still differing per item. Uniform columns hide bugs in
    # the paired code, which joins per item and would look fine if every value
    # matched.
    LOW, SPAN = 0.85, 0.15

    def __init__(self, judge: FakeJudge | None = None):
        self.judge = judge or FakeJudge()

    def score(self, question: str, answer: str, contexts: list[str]) -> ItemScores:
        seed = self.judge.seed
        # Relevance needs only the question and the answer, which is exactly how
        # the real metrics behave -- and why an app that stops retrieving can
        # still score 0.988 on relevance while being grounded in nothing.
        relevance = round(self.LOW + self.SPAN * _unit(seed, "rel", question, answer), 4)

        if not contexts:
            # What a real Scorer returns when the target retrieved nothing:
            # faithfulness is UNMEASURABLE, not zero. A fake that also dropped
            # relevance here would hide the very asymmetry that makes the
            # "fluent but ungrounded" failure detectable.
            return ItemScores(
                faithfulness=None,
                answer_relevancy=relevance,
                errors={"faithfulness": "fake judge: no contexts to check against"},
            )

        supported, total = _grounding(answer, contexts)
        return ItemScores(
            faithfulness=1.0 if total == 0 else round(supported / total, 4),
            answer_relevancy=relevance,
            reasons={
                "faithfulness": (
                    f"fake judge: {supported}/{total} sentence(s) found verbatim in "
                    f"the retrieved passages"
                ),
                "answer_relevancy": "fake judge: synthetic score, no model was called",
            },
        )
