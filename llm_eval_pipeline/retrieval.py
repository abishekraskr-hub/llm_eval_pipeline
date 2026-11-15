"""Retrieval metrics: score the search step on its own, with no LLM calls.

Why this exists. Grading only the answer leaves a blind spot the platform already
walked into: when retrieval broke completely, `answer_relevance` still scored
0.988, because the answers stayed fluent and on-topic -- they were just grounded
in nothing. Only the "a declared metric that could not be measured FAILS" rule
caught it, and that is a backstop, not a measurement.

These three metrics measure retrieval directly against human-labelled gold
locations. They are deterministic, cost nothing, and move the moment retrieval
regresses -- before any judge is involved.
"""
from dataclasses import dataclass

from .targets.base import RetrievedSource

# A location is (source_id, page). `page` may be None, which is what lets this
# work outside document retrieval: an app querying OpenSearch or a SQL table has
# a stable source_id and no page at all.
Location = tuple[str, int | None]


@dataclass
class RetrievalScores:
    """Per-item retrieval scores. `None` means not measured, never 0.0.

    The distinction matters: 0.0 says "we looked and retrieval found nothing
    relevant", while None says "this target does not report retrieval, so there
    was nothing to look at". The gate treats the second as a failure to measure.
    """

    hit: float | None = None
    recall: float | None = None
    rr: float | None = None

    @property
    def measured(self) -> bool:
        return self.hit is not None


def matches(retrieved: Location, gold: Location, page_tolerance: int = 0) -> bool:
    """Does a retrieved location satisfy a gold one?

    Exact on `source_id`. The page is compared **only when both sides declare
    one**, within `page_tolerance`.

    That rule is what keeps page-less retrieval usable. If a missing page were
    treated as a mismatch, every app without page numbers would score 0 on every
    retrieval metric; if it were treated as a wildcard in one direction only, the
    metric would mean different things depending on which side omitted it. So:
    both present -> compare; either absent -> the source_id match stands alone.

    Tolerance exists because a chunk can straddle a page boundary -- the answer
    labelled on page 12 may sit in a chunk the index stamped as page 11. Default
    0: widen it only once you have actually seen that happen, since every page of
    slack makes the metric easier to pass.
    """
    if retrieved[0] != gold[0]:
        return False
    if retrieved[1] is None or gold[1] is None:
        return True
    return abs(retrieved[1] - gold[1]) <= page_tolerance


def score_retrieval(
    gold: list[Location],
    retrieved: list[RetrievedSource] | None,
    k: int = 5,
    page_tolerance: int = 0,
) -> RetrievalScores:
    """Score one item's retrieval against its gold locations.

    hit    -- 1.0 if any gold location appears in the top k, else 0.0
    recall -- fraction of gold locations that appear in the top k
    rr     -- 1 / rank of the first gold page found, 0.0 if none (reciprocal rank;
              the run-level mean of this is MRR)
    """
    if retrieved is None:
        return RetrievalScores()  # all None: not measured
    if not gold:
        # No gold labels means nothing to score against. Callers only pass
        # answerable items, and the dataset loader rejects those without labels,
        # so this is a guard rather than an expected path.
        return RetrievalScores()

    top = [r for r in retrieved if r.rank <= k]

    found = sum(
        1
        for g in gold
        if any(matches((r.source_id, r.page), g, page_tolerance) for r in top)
    )
    recall = found / len(gold)

    rr = 0.0
    for r in sorted(top, key=lambda r: r.rank):
        if any(matches((r.source_id, r.page), g, page_tolerance) for g in gold):
            rr = 1.0 / r.rank
            break

    return RetrievalScores(hit=1.0 if found else 0.0, recall=recall, rr=rr)


def metric_names(k: int) -> dict[str, str]:
    """The summary keys these metrics land under, for a given k.

    `k` is baked into the name because a hit rate at 5 and a hit rate at 20 are
    different measurements, and a run history that mixes them silently would be
    worse than useless. `retrieval_mrr` carries no k: it is defined over the whole
    ranked list the app returned, truncated at k like the others, but its value is
    dominated by the top few ranks either way.
    """
    return {
        "hit": f"retrieval_hit_rate@{k}",
        "recall": f"retrieval_recall@{k}",
        "rr": "retrieval_mrr",
    }


def resolve_metric_key(name: str, k: int) -> str:
    """Let config say `retrieval_hit_rate@k` and mean `@5` when k is 5.

    Writing the literal k in eval_config.yaml would mean two places to edit for one
    change, and the two would drift. `@k` is the template; `@5` is also accepted so
    an explicit name still works.
    """
    return name.replace("@k", f"@{k}")
