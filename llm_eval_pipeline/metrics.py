"""Scoring one answer.

A note on metric direction, because the names mislead:

DeepEval's `HallucinationMetric` scores the *fraction of contexts the answer
agrees with* -- higher is better -- and its prompt says to answer "no" ONLY on a
direct CONTRADICTION. So it does not flag an unsupported *addition*: given the
context "Drug X causes nausea" and the answer "Drug X causes nausea and kidney
failure", it reports agreement. That is exactly the failure a hallucination gate
is supposed to catch.

`FaithfulnessMetric` is the one that catches it: it extracts the claims in the
answer and checks each against the retrieved context, so an unsupported addition
lowers the score. We therefore derive the project's "hallucination rate" from
faithfulness, not from the metric that happens to be named after it.

Every score here is 0..1 and higher-is-better. The gate converts to rates.
"""
from dataclasses import dataclass, field

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from .judge import JudgeQuotaExceeded, LLMJudge

# An answer is counted as hallucinating when its faithfulness falls below this.
# 1.0 would mean "every single extracted claim is supported"; judges are noisy
# enough that a hair of slack avoids flagging well-grounded answers.
FAITHFULNESS_CLEAN = 0.99


@dataclass
class ItemScores:
    """Per-question scores. `None` means not measured, which is not the same as 0."""

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    reasons: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def hallucinated(self) -> bool | None:
        if self.faithfulness is None:
            return None
        return self.faithfulness < FAITHFULNESS_CLEAN


class Scorer:
    def __init__(
        self,
        judge: LLMJudge,
        measure_faithfulness: bool = True,
        measure_relevancy: bool = True,
    ):
        self.judge = judge
        self.measure_faithfulness = measure_faithfulness
        self.measure_relevancy = measure_relevancy

    def _metrics(self) -> list[tuple[str, object]]:
        built: list[tuple[str, object]] = []
        # async_mode=False on purpose: DeepEval would otherwise fan concurrent
        # judge calls out and blow the per-minute quota the judge is pacing.
        if self.measure_faithfulness:
            built.append(
                ("faithfulness", FaithfulnessMetric(model=self.judge, async_mode=False, threshold=0.0))
            )
        if self.measure_relevancy:
            built.append(
                (
                    "answer_relevancy",
                    AnswerRelevancyMetric(model=self.judge, async_mode=False, threshold=0.0),
                )
            )
        return built

    def score(self, question: str, answer: str, contexts: list[str]) -> ItemScores:
        """Score one answer. Thresholds live in the gate, so metrics are set to 0.0
        here -- we want the raw number, not DeepEval's own pass/fail."""
        scores = ItemScores()
        if not answer.strip():
            scores.errors["all"] = "empty answer"
            return scores

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=list(contexts) if contexts else None,
        )

        for name, metric in self._metrics():
            if name == "faithfulness" and not contexts:
                # Faithfulness is undefined without retrieved context; recording 0
                # would be a silent lie about a measurement we never made.
                scores.errors[name] = "no retrieval context returned by target"
                continue
            try:
                metric.measure(test_case)
                setattr(scores, name, float(metric.score))
                if getattr(metric, "reason", None):
                    scores.reasons[name] = str(metric.reason)
            except JudgeQuotaExceeded:
                raise  # a quota wall is a run-level problem, not a bad score
            except Exception as exc:
                scores.errors[name] = f"{type(exc).__name__}: {exc}"
        return scores
