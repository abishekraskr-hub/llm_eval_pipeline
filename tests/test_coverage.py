"""The empty-answer loophole, and the two rules that close it.

This file exists because of a real run that PASSED every quality check while a
fifth of the dataset went unanswered. The app returned `""` for one question in
five. That answer was:

* not a refusal, so `abstention_rate` ignored it;
* not a wrong answer, so `false_refusal_rate` ignored it;
* ungradeable, so the judge returned None for faithfulness and relevance -- and
  None is EXCLUDED from the mean, so the item did not drag the average down, it
  vanished from it.

The reported faithfulness was 1.000: the mean of the four questions that worked.
`require_all_metrics` did not help, because it asks "was this metric measured at
all" and the other four items had measured it.

Two rules close it, and they are deliberately independent -- one needs no judge,
the other is general:

1. `answered_rate`, a metric that needs no LLM and applies to every item, so it
   cannot itself go missing;
2. per-metric COVERAGE: a metric measured on only part of the items it applies
   to fails, however good the average over that part looks.
"""
import pytest

from conftest import REFUSAL_POLICY
from llm_eval_pipeline.aggregate import (
    has_answer,
    item_metric_applicable,
    metric_coverage,
    summarize,
)
from llm_eval_pipeline.gate import evaluate
from llm_eval_pipeline.store import METRIC_COLUMNS, Store

# The run that exposed this, reduced to its essentials: three answerable
# questions, one of which came back empty, and two unanswerable ones that were
# correctly refused.
REAL_RUN = [
    {
        "id": "a1",
        "kind": "answerable",
        "answer": "",  # <- the whole problem
        "contexts": ["ctx"],
        "gold_sources": [["f.pdf", 1]],
        "retrieval": {"hit": 1.0, "recall": 1.0, "rr": 0.5},
        "latency_s": 6.7,
        "metric_errors": {"all": "empty answer"},
    },
    {
        "id": "a2",
        "kind": "answerable",
        "answer": "Grounded, cited text [f.pdf p.1].",
        "faithfulness": 1.0,
        "answer_relevancy": 1.0,
        "hallucinated": False,
        "contexts": ["ctx"],
        "gold_sources": [["f.pdf", 1]],
        "retrieval": {"hit": 1.0, "recall": 1.0, "rr": 1.0},
        "latency_s": 3.7,
    },
    {
        "id": "a3",
        "kind": "answerable",
        "answer": "Grounded, cited text [f.pdf p.1].",
        "faithfulness": 1.0,
        "answer_relevancy": 1.0,
        "hallucinated": False,
        "contexts": ["ctx"],
        "gold_sources": [["f.pdf", 1]],
        "retrieval": {"hit": 1.0, "recall": 1.0, "rr": 1.0},
        "latency_s": 13.5,
    },
    {
        "id": "u1",
        "kind": "unanswerable",
        "answer": "This is not mentioned in the available documents.",
        "faithfulness": 1.0,
        "answer_relevancy": 1.0,
        "hallucinated": False,
        "contexts": ["ctx"],
        "latency_s": 9.8,
    },
    {
        "id": "u2",
        "kind": "unanswerable",
        "answer": "This is not mentioned in the available documents.",
        "faithfulness": 1.0,
        "answer_relevancy": 1.0,
        "hallucinated": False,
        "contexts": ["ctx"],
        "latency_s": 12.6,
    },
]

THRESHOLDS = {
    "answered_rate": {"floor": 1.0, "type": "binary", "direction": "higher_is_better"},
    "faithfulness": {"floor": 0.75, "type": "continuous", "direction": "higher_is_better"},
    "answer_relevance": {"floor": 0.75, "type": "continuous", "direction": "higher_is_better"},
    "abstention_rate": {"floor": 0.70, "type": "binary", "direction": "higher_is_better"},
    "false_refusal_rate": {"floor": 0.20, "type": "binary", "direction": "lower_is_better"},
    "retrieval_hit_rate@k": {"floor": 0.70, "type": "binary", "direction": "higher_is_better"},
    "latency_p95": 20.0,
}


def gate_for(results, **kwargs):
    summary = summarize(results, retrieval_k=5, refusal=REFUSAL_POLICY).to_dict()
    return summary, evaluate(summary, THRESHOLDS, retrieval_k=5, **kwargs)


class TestTheLoopholeItself:
    def test_an_empty_answer_still_flatters_the_average(self):
        """Documenting the hole, not defending it.

        This is why neither faithfulness nor relevance can be the thing that
        catches an unanswered question: both look perfect.
        """
        summary, _ = gate_for(REAL_RUN)
        assert summary["faithfulness"] == 1.0
        assert summary["answer_relevance"] == 1.0
        assert summary["hallucination_rate"] == 0.0

    def test_the_refusal_pair_does_not_see_it_either(self):
        """An empty string is not a refusal, so it scores as a non-refusal."""
        summary, _ = gate_for(REAL_RUN)
        assert summary["abstention_rate"] == 1.0
        assert summary["false_refusal_rate"] == 0.0

    def test_but_the_gate_now_fails(self):
        """The point of the whole file."""
        _, gate = gate_for(REAL_RUN)
        assert gate.passed is False

    def test_and_it_fails_for_the_right_reason(self):
        """A gate that fails for the wrong reason teaches people to ignore it.

        The original run failed on p95 latency, which was a mis-set threshold and
        had nothing to do with the actual defect.
        """
        _, gate = gate_for(REAL_RUN)
        failures = {c.metric for c in gate.failures}
        assert failures == {"answered_rate"}

    def test_the_unanswered_item_is_named(self):
        """So the reader does not have to diff five answers to find it."""
        summary, _ = gate_for(REAL_RUN)
        assert summary["unanswered"] == ["a1"]
        assert any("a1" in note for note in summary["notes"])


class TestAnsweredRate:
    def test_it_is_the_fraction_that_produced_text(self):
        summary, _ = gate_for(REAL_RUN)
        assert summary["answered_rate"] == pytest.approx(0.8)

    def test_a_full_run_scores_one(self):
        good = [r for r in REAL_RUN if r["id"] != "a1"]
        summary, gate = gate_for(good)
        assert summary["answered_rate"] == 1.0
        assert gate.passed is True

    def test_whitespace_is_not_an_answer(self):
        assert has_answer({"answer": "   \n\t "}) is False
        assert has_answer({"answer": "a"}) is True

    def test_an_errored_item_counts_as_unanswered(self):
        """Both failures leave the question unanswered, so both must pull the
        same metric down. Otherwise an outage reads as a smaller dataset."""
        assert has_answer({"answer": "text", "error": "target unavailable"}) is False

    def test_it_needs_no_judge_so_it_cannot_go_missing(self):
        """The reason this metric exists at all: every judge-scored metric can
        return None, and None disappears from the mean."""
        unscored = [
            {"id": "a1", "kind": "answerable", "answer": "text", "latency_s": 1.0},
            {"id": "a2", "kind": "answerable", "answer": "", "latency_s": 1.0},
        ]
        summary = summarize(unscored, refusal=REFUSAL_POLICY).to_dict()
        assert summary["faithfulness"] is None, "no judge ran"
        assert summary["answered_rate"] == 0.5, "but this is still measured"


class TestApplicability:
    """A None per-item value means two different things. Only one is a fault."""

    def test_abstention_applies_only_to_unanswerable_items(self):
        applies = item_metric_applicable(REAL_RUN, retrieval_k=5, refusal=REFUSAL_POLICY)
        assert applies["abstention_rate"] == {"u1", "u2"}

    def test_false_refusal_applies_only_to_answerable_items(self):
        applies = item_metric_applicable(REAL_RUN, retrieval_k=5, refusal=REFUSAL_POLICY)
        # a1 is excluded: it produced no answer, so it is answered_rate's
        # business, not the refusal pair's.
        assert applies["false_refusal_rate"] == {"a2", "a3"}

    def test_retrieval_applies_only_where_a_gold_label_exists(self):
        """An answerable question with no gold_sources is an UNLABELLED question,
        not a retrieval failure. Counting it as one would punish the retriever
        for a gap in the dataset."""
        results = [
            {"id": "a1", "kind": "answerable", "answer": "x", "gold_sources": [["f.pdf", 1]]},
            {"id": "a2", "kind": "answerable", "answer": "x", "gold_sources": []},
            {"id": "u1", "kind": "unanswerable", "answer": "x"},
        ]
        applies = item_metric_applicable(results, retrieval_k=5, refusal=REFUSAL_POLICY)
        assert applies["retrieval_hit_rate@5"] == {"a1"}

    def test_an_unanswered_item_is_not_applicable_to_the_judge_metrics(self):
        """There is nothing to grade, so holding faithfulness responsible would
        double-count the same defect and blame the wrong component."""
        applies = item_metric_applicable(REAL_RUN, retrieval_k=5, refusal=REFUSAL_POLICY)
        assert "a1" not in applies["faithfulness"]
        assert "a1" in applies["answered_rate"]


class TestCoverage:
    def test_retrieval_still_counts_the_unanswered_item(self):
        """Retrieval is scored BEFORE the judge, so an empty answer does not
        hide a working retriever -- and this is what makes the gate's attribution
        correct: retrieval passed, answering failed."""
        coverage = metric_coverage(REAL_RUN, retrieval_k=5, refusal=REFUSAL_POLICY)
        assert coverage["retrieval_hit_rate@5"] == {"measured": 3, "applicable": 3}

    def test_a_metric_measured_on_part_of_the_dataset_fails(self):
        """The general rule. Here the judge errored on one answerable item of
        three, leaving a 1.000 average over the two that worked."""
        results = [
            dict(r) for r in REAL_RUN if r["id"] != "a1"
        ] + [
            {
                "id": "a4",
                "kind": "answerable",
                "answer": "an answer the judge could not score",
                "contexts": ["ctx"],
                "gold_sources": [["f.pdf", 1]],
                "retrieval": {"hit": 1.0, "recall": 1.0, "rr": 1.0},
                "latency_s": 2.0,
                "metric_errors": {"faithfulness": "judge timed out"},
            }
        ]
        summary, gate = gate_for(results, min_metric_coverage=0.9)
        assert summary["faithfulness"] == 1.0, "the average over what was scored"
        check = next(c for c in gate.checks if c.metric == "faithfulness")
        assert check.partial_coverage is True
        assert check.passed is False
        assert "4/5" in check.failure_reason
        assert gate.passed is False

    def test_the_reason_says_how_many_items_were_measured(self):
        """A bare FAIL on a metric reading 1.000 is baffling. The count is the
        explanation."""
        results = [dict(r) for r in REAL_RUN if r["id"] != "a1"] + [
            {
                "id": "a4",
                "kind": "answerable",
                "answer": "unscored",
                "contexts": ["ctx"],
                "latency_s": 2.0,
            }
        ]
        _, gate = gate_for(results, min_metric_coverage=1.0)
        check = next(c for c in gate.checks if c.metric == "faithfulness")
        assert check.failure_reason == "measured on only 4/5 items"
        assert check.coverage == pytest.approx(0.8)

    def test_full_coverage_passes(self):
        good = [r for r in REAL_RUN if r["id"] != "a1"]
        _, gate = gate_for(good, min_metric_coverage=1.0)
        assert gate.passed is True
        for check in gate.checks:
            assert check.partial_coverage is False

    def test_aggregate_metrics_are_exempt(self):
        """p95 has no per-item values at all, so a coverage rule would fail it
        permanently."""
        _, gate = gate_for(REAL_RUN, min_metric_coverage=1.0)
        check = next(c for c in gate.checks if c.metric == "latency_p95")
        assert check.partial_coverage is False
        assert check.passed is True

    def test_zero_switches_the_check_off(self):
        """An escape hatch that must not also switch off answered_rate, which is
        a floor rather than a coverage rule."""
        _, gate = gate_for(REAL_RUN, min_metric_coverage=0.0)
        assert all(not c.partial_coverage for c in gate.checks)
        assert gate.passed is False, "answered_rate still fails"

    def test_coverage_is_reported_for_every_metric_not_just_failing_ones(self):
        """A reader who can see 4/4 next to a score knows it covers the dataset.
        Without it, a flattering subset average looks identical to an honest one.
        """
        _, gate = gate_for(REAL_RUN)
        scored = [c for c in gate.checks if c.metric != "latency_p95"]
        assert scored and all(c.applicable for c in scored)


class TestItIsPersisted:
    def test_answered_rate_has_a_column(self):
        """It is a per-item metric like any other, so it has to survive into the
        store -- otherwise it could not be a baseline for the paired test."""
        assert "answered_rate" in METRIC_COLUMNS

    def test_it_round_trips(self, tmp_path):
        store = Store(tmp_path / "x.db")
        run_id = store.start_run(started_at="t", target={})
        store.add_result(run_id, {"id": "a1", "kind": "answerable"}, {"answered_rate": 0.0})
        store.finish_run(
            run_id, finished_at="t2", summary={}, gate_passed=False, gate_detail=[]
        )
        assert store.item_values(run_id) == {"answered_rate": {"a1": 0.0}}

    def test_an_old_database_gains_the_column(self, tmp_path):
        """Adding a metric must not require deleting run history."""
        import sqlite3

        path = tmp_path / "old.db"
        with sqlite3.connect(path) as conn:
            conn.executescript(
                """
                CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                   started_at TEXT NOT NULL);
                CREATE TABLE results (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                      run_id INTEGER NOT NULL, item_id TEXT NOT NULL,
                                      faithfulness REAL);
                INSERT INTO runs (started_at) VALUES ('t');
                INSERT INTO results (run_id, item_id, faithfulness) VALUES (1, 'a1', 0.9);
                """
            )
        store = Store(path)
        rows = store.metrics_for(1)
        assert "answered_rate" in rows[0]
        assert rows[0]["faithfulness"] == 0.9, "existing scores kept"
