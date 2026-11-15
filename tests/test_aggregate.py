"""Aggregation: the refusal pair, and the summary-equals-mean-of-items invariant."""
import pytest

from conftest import REFUSAL_POLICY
from llm_eval_pipeline.aggregate import item_metric_values, summarize

REFUSAL = "This is not mentioned in the available documents."
ANSWER = "Omeprazole may cause headache and abdominal pain [doc.pdf p.4]."


def record(item_id, kind, answer, faithfulness=1.0, relevancy=0.97, hallucinated=False):
    return {
        "id": item_id,
        "kind": kind,
        "answer": answer,
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "hallucinated": hallucinated,
        "latency_s": 1.0,
    }


class TestRefusalDetector:
    @pytest.mark.parametrize(
        "text",
        [
            "This is not mentioned in the available documents.",
            "The documents do not mention hair loss.",
            "No information is available on that.",
            "The dose is NOT SPECIFIED in these papers.",
            "I am unable to answer from the provided sources.",
        ],
    )
    def test_detects_refusals(self, text):
        assert REFUSAL_POLICY.verdict({'answer': text}) is True

    @pytest.mark.parametrize("text", [ANSWER, "", "Yes, it causes headache."])
    def test_ignores_real_answers(self, text):
        assert REFUSAL_POLICY.verdict({'answer': text}) is False

    def test_shared_by_both_metrics(self):
        """One detector: abstention and false-refusal can never disagree."""
        hedged = "It may help, though this is not mentioned in detail."
        values = item_metric_values(
            [record("u1", "unanswerable", hedged), record("a1", "answerable", hedged)],
            refusal=REFUSAL_POLICY,
        )
        assert values["abstention_rate"]["u1"] == 1.0
        assert values["false_refusal_rate"]["a1"] == 1.0


class TestRefusalMetrics:
    def test_refusing_everything_scores_perfectly_and_terribly(self):
        """The failure abstention_rate alone cannot see."""
        results = [
            record("a1", "answerable", REFUSAL),
            record("a2", "answerable", REFUSAL),
            record("u1", "unanswerable", REFUSAL),
        ]
        summary = summarize(results, refusal=REFUSAL_POLICY).to_dict()
        assert summary["abstention_rate"] == 1.0  # looks perfect
        assert summary["false_refusal_rate"] == 1.0  # and is catastrophic

    def test_answering_everything_is_the_mirror_image(self):
        results = [
            record("a1", "answerable", ANSWER),
            record("u1", "unanswerable", ANSWER),
            record("u2", "unanswerable", ANSWER),
        ]
        summary = summarize(results, refusal=REFUSAL_POLICY).to_dict()
        assert summary["abstention_rate"] == 0.0
        assert summary["false_refusal_rate"] == 0.0

    def test_denominators_are_separate(self):
        """Each rate is over its own kind of item only."""
        results = [
            record("a1", "answerable", ANSWER),
            record("a2", "answerable", REFUSAL),
            record("u1", "unanswerable", REFUSAL),
            record("u2", "unanswerable", ANSWER),
        ]
        summary = summarize(results, refusal=REFUSAL_POLICY).to_dict()
        assert summary["answerable_total"] == 2
        assert summary["unanswerable_total"] == 2
        assert summary["false_refusal_rate"] == 0.5
        assert summary["abstention_rate"] == 0.5

    def test_missing_kind_is_noted_not_zeroed(self):
        summary = summarize([record("a1", "answerable", ANSWER)], refusal=REFUSAL_POLICY).to_dict()
        assert summary["abstention_rate"] is None
        assert any("abstention_rate not measured" in n for n in summary["notes"])

    def test_limit_warning_mentions_the_slicing_trap(self):
        summary = summarize([record("a1", "answerable", ANSWER)], refusal=REFUSAL_POLICY).to_dict()
        note = next(n for n in summary["notes"] if "abstention_rate" in n)
        assert "FIRST n" in note

    def test_errored_items_are_excluded_from_both(self):
        results = [record("a1", "answerable", ANSWER)]
        results.append({"id": "a2", "kind": "answerable", "error": "boom"})
        values = item_metric_values(results)
        assert values["false_refusal_rate"]["a2"] is None


class TestSummaryIsMeanOfItems:
    """The invariant the paired gate depends on."""

    @pytest.mark.parametrize(
        "metric",
        [
            "faithfulness",
            "answer_relevance",
            "hallucination_rate",
            "abstention_rate",
            "false_refusal_rate",
        ],
    )
    def test_every_rate_equals_its_item_mean(self, metric):
        results = [
            record("a1", "answerable", ANSWER, faithfulness=0.80, hallucinated=True),
            record("a2", "answerable", REFUSAL, faithfulness=1.00, hallucinated=False),
            record("a3", "answerable", ANSWER, faithfulness=0.95, hallucinated=True),
            record("u1", "unanswerable", REFUSAL, faithfulness=1.00, hallucinated=False),
            record("u2", "unanswerable", ANSWER, faithfulness=1.00, hallucinated=False),
        ]
        summary = summarize(results, refusal=REFUSAL_POLICY).to_dict()
        values = [
            v
            for v in item_metric_values(results, refusal=REFUSAL_POLICY)[metric].values()
            if v is not None
        ]
        expected = sum(values) / len(values)
        assert summary[metric] == pytest.approx(expected, abs=1e-4)

    def test_holds_for_retrieval_too(self):
        results = [
            {
                "id": f"a{i}",
                "kind": "answerable",
                "answer": ANSWER,
                "latency_s": 1.0,
                "retrieval": {"hit": hit, "recall": rec, "rr": rr},
            }
            for i, (hit, rec, rr) in enumerate([(1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.5, 0.5)])
        ]
        summary = summarize(results, retrieval_k=5).to_dict()
        values = item_metric_values(results, 5)
        for key in ("retrieval_hit_rate@5", "retrieval_recall@5", "retrieval_mrr"):
            measured = [v for v in values[key].values() if v is not None]
            assert summary[key] == pytest.approx(sum(measured) / len(measured), abs=1e-4)


class TestLatency:
    def test_percentiles_on_odd_n(self):
        """9 items: the median is unambiguous, whichever nearest-rank convention."""
        results = [record(f"a{i}", "answerable", ANSWER) for i in range(9)]
        for i, r in enumerate(results):
            r["latency_s"] = float(i + 1)
        summary = summarize(results, refusal=REFUSAL_POLICY).to_dict()
        assert summary["latency_p50"] == 5.0
        assert summary["latency_p95"] == 9.0

    def test_even_n_takes_the_upper_middle(self):
        """Documents the existing convention rather than asserting a preference.

        Nearest-rank has several definitions that disagree at exact boundaries;
        this one rounds half up, so p50 of 1..10 is 6, not 5. It matters only that
        it is consistent between runs, which is what the gate compares.
        """
        results = [record(f"a{i}", "answerable", ANSWER) for i in range(10)]
        for i, r in enumerate(results):
            r["latency_s"] = float(i + 1)
        summary = summarize(results, refusal=REFUSAL_POLICY).to_dict()
        assert summary["latency_p50"] == 6.0
        assert summary["latency_p95"] == 10.0

    def test_single_item(self):
        results = [record("a1", "answerable", ANSWER)]
        results[0]["latency_s"] = 4.2
        summary = summarize(results, refusal=REFUSAL_POLICY).to_dict()
        assert summary["latency_p50"] == summary["latency_p95"] == 4.2
