"""Shared stubs. Nothing here touches the network or needs an API key.

The point of these is that the gate and the statistics can be tested without
spending quota. A test suite that needed a live judge would be run once and then
never again.
"""
import sys
from pathlib import Path

import pytest

# Make `llm_eval_pipeline` importable however pytest was invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_eval_pipeline.aggregate import RefusalPolicy  # noqa: E402
from llm_eval_pipeline.dataset import GoldenItem  # noqa: E402

# A refusal policy for tests. The library ships none on purpose -- it cannot know
# how a given app phrases "I don't know" -- so every test that exercises
# abstention_rate or false_refusal_rate has to supply one, exactly as a real app
# does in its llmeval.yaml.
REFUSAL_POLICY = RefusalPolicy.from_config(
    {
        "refusal": {
            "use_target_flag": True,
            "patterns": [
                r"not mentioned",
                r"do(es)? not mention",
                r"not contain",
                r"no information",
                r"not specified",
                r"not available in",
                r"not found in",
                r"cannot answer",
                r"unable to answer",
                r"not discussed",
                r"not provided",
                r"not covered",
            ],
        }
    }
)
from llm_eval_pipeline.metrics import ItemScores  # noqa: E402


class StubJudge:
    """Satisfies the two things the runner asks of a judge."""

    def __init__(self, name: str = "stub:judge"):
        self._name = name
        self.calls = 0

    def get_model_name(self) -> str:
        return self._name


class StubScorer:
    """Returns fixed scores, so a test exercises the gate rather than a model.

    `faithfulness=None` is the interesting case: it is what a real Scorer returns
    when the target handed back no contexts, and the gate must FAIL on it.
    """

    def __init__(self, faithfulness=1.0, relevancy=0.97, per_item: dict | None = None):
        self.faithfulness = faithfulness
        self.relevancy = relevancy
        self.per_item = per_item or {}
        self.calls = 0

    def score(self, question: str, answer: str, contexts: list[str]) -> ItemScores:
        self.calls += 1
        override = self.per_item.get(question, {})
        faith = override.get("faithfulness", self.faithfulness)
        rel = override.get("answer_relevancy", self.relevancy)
        scores = ItemScores(faithfulness=faith, answer_relevancy=rel)
        if faith is None:
            scores.errors["faithfulness"] = "no retrieval context returned by target"
        return scores


def item(
    item_id: str,
    question: str | None = None,
    kind: str = "answerable",
    gold=((  "doc.pdf", 1),),
):
    """A GoldenItem with sane defaults; unanswerable items get no gold sources."""
    return GoldenItem(
        id=item_id,
        question=question or f"question {item_id}",
        kind=kind,
        gold_sources=[] if kind == "unanswerable" else [tuple(g) for g in gold],
    )


@pytest.fixture
def mixed_items():
    """Three answerable, two unanswerable -- enough for both refusal metrics."""
    return [
        item("a1"),
        item("a2"),
        item("a3"),
        item("u1", kind="unanswerable"),
        item("u2", kind="unanswerable"),
    ]
