"""Retrieval metrics, including the worked example from the spec."""
import pytest

from llm_eval_pipeline.aggregate import summarize
from llm_eval_pipeline.retrieval import metric_names, resolve_metric_key, score_retrieval
from llm_eval_pipeline.targets.base import RetrievedSource, rank_sources

DOC = "doc.pdf"


def ranked(*pages: int) -> list[RetrievedSource]:
    return [RetrievedSource(file=DOC, page=p, rank=i) for i, p in enumerate(pages, start=1)]


# The spec's table, verbatim. k = 3.
#   Item | Gold    | Retrieved      | hit | recall | RR
#   A    | p12     | p12, p13, p40  | 1   | 1.0    | 1.0
#   B    | p5      | p22, p5,  p8   | 1   | 1.0    | 0.5
#   C    | p7, p9  | p7,  p30, p31  | 1   | 0.5    | 1.0
#   D    | p3      | p22, p31, p8   | 0   | 0.0    | 0.0
WORKED_EXAMPLE = [
    ("A", [(DOC, 12)], ranked(12, 13, 40), 1.0, 1.0, 1.0),
    ("B", [(DOC, 5)], ranked(22, 5, 8), 1.0, 1.0, 0.5),
    ("C", [(DOC, 7), (DOC, 9)], ranked(7, 30, 31), 1.0, 0.5, 1.0),
    ("D", [(DOC, 3)], ranked(22, 31, 8), 0.0, 0.0, 0.0),
]


@pytest.mark.parametrize("name,gold,retrieved,hit,recall,rr", WORKED_EXAMPLE)
def test_worked_example_per_item(name, gold, retrieved, hit, recall, rr):
    scores = score_retrieval(gold, retrieved, k=3)
    assert scores.hit == hit, name
    assert scores.recall == pytest.approx(recall), name
    assert scores.rr == pytest.approx(rr), name


def test_worked_example_aggregates():
    """hit_rate = 0.75, recall = 0.625, MRR = 0.625."""
    results = []
    for name, gold, retrieved, *_ in WORKED_EXAMPLE:
        scores = score_retrieval(gold, retrieved, k=3)
        results.append(
            {
                "id": name,
                "kind": "answerable",
                "answer": "an answer that is not a refusal",
                "retrieval": {"hit": scores.hit, "recall": scores.recall, "rr": scores.rr},
            }
        )
    summary = summarize(results, retrieval_k=3).to_dict()
    assert summary["retrieval_hit_rate@3"] == pytest.approx(0.75)
    assert summary["retrieval_recall@3"] == pytest.approx(0.625)
    assert summary["retrieval_mrr"] == pytest.approx(0.625)
    assert summary["retrieval_scored"] == 4


def test_unmeasured_when_target_reports_no_retrieval():
    """None must survive as None -- it is not a zero score."""
    scores = score_retrieval([(DOC, 1)], None, k=5)
    assert scores.hit is None and scores.recall is None and scores.rr is None
    assert not scores.measured


def test_empty_retrieval_is_a_measured_zero():
    """An app that exposes retrieval but found nothing scores 0, not None."""
    scores = score_retrieval([(DOC, 1)], [], k=5)
    assert (scores.hit, scores.recall, scores.rr) == (0.0, 0.0, 0.0)
    assert scores.measured


def test_k_truncates():
    """A gold hit below the cut-off does not count."""
    assert score_retrieval([(DOC, 9)], ranked(1, 2, 9), k=2).hit == 0.0
    assert score_retrieval([(DOC, 9)], ranked(1, 2, 9), k=3).hit == 1.0


def test_page_tolerance():
    gold = [(DOC, 12)]
    assert score_retrieval(gold, ranked(11), k=5).hit == 0.0
    assert score_retrieval(gold, ranked(11), k=5, page_tolerance=1).hit == 1.0
    assert score_retrieval(gold, ranked(14), k=5, page_tolerance=1).hit == 0.0


def test_filename_must_match_exactly():
    other = [RetrievedSource(file="other.pdf", page=12, rank=1)]
    assert score_retrieval([(DOC, 12)], other, k=5, page_tolerance=5).hit == 0.0


class TestRankSources:
    def test_union_across_calls_keeps_first_appearance(self):
        """An agentic app searches more than once; the first search ranks higher."""
        sources = [
            {"source": "a.pdf", "page": 3, "score": 0.4},  # first search
            {"source": "b.pdf", "page": 1, "score": 0.9},
            {"source": "a.pdf", "page": 3, "score": 0.95},  # second search, duplicate
            {"source": "c.pdf", "page": 7, "score": 0.5},
        ]
        got = rank_sources(sources)
        assert [(r.file, r.page, r.rank) for r in got] == [
            ("a.pdf", 3, 1),
            ("b.pdf", 1, 2),
            ("c.pdf", 7, 3),
        ]

    def test_skips_hits_with_no_source_id(self):
        """A hit that cannot be identified cannot be matched against a gold
        label, so counting it would only dilute the metric. A hit with an id and
        NO page is fine, though -- that is what page-less retrieval looks like."""
        assert rank_sources([{"score": 0.9}, {"page": 2}]) == []

        identified = rank_sources([{"source": "a.pdf"}, {"source_id": "doc-7"}])
        assert [(r.source_id, r.page, r.rank) for r in identified] == [
            ("a.pdf", None, 1),
            ("doc-7", None, 2),
        ]

    def test_accepts_either_key_spelling(self):
        got = rank_sources([{"file": "a.pdf", "page": 1}])
        assert (got[0].file, got[0].rank) == ("a.pdf", 1)


def test_metric_names_carry_k():
    assert metric_names(5)["hit"] == "retrieval_hit_rate@5"
    assert metric_names(20)["recall"] == "retrieval_recall@20"
    assert metric_names(5)["rr"] == "retrieval_mrr"


def test_resolve_metric_key_template():
    assert resolve_metric_key("retrieval_hit_rate@k", 5) == "retrieval_hit_rate@5"
    assert resolve_metric_key("retrieval_mrr", 5) == "retrieval_mrr"
