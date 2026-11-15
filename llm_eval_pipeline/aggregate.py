"""Turn per-question results into the run-level numbers the gate checks.

One invariant holds for every metric here: **the run-level value is exactly the
mean of that metric's per-item values**, where `None` means "not applicable or not
measured for this item" and is excluded from the mean. Rates included -- an
abstention rate is the mean of a 0/1 "did it refuse" over unanswerable items.

Excluding `None` from the mean has a sharp edge, and `coverage` exists because of
it: an item the judge could not score DISAPPEARS from the average rather than
dragging it down. A run where the app returned an empty answer for one question
in three therefore reported faithfulness 1.000 -- the mean of the two that
worked. `coverage` records, per metric, how many items it was measured on versus
how many it applies to, so the gate can fail a metric that quietly stopped being
measured on part of the dataset. `answered_rate` is the blunt companion to it: it
needs no judge, so it can never go missing itself.

That is not a stylistic choice. The paired statistical gate compares a PR to a
baseline item by item, and its verdict must be about the same quantity the
absolute floor is about. If the summary computed a rate one way and the paired
test another, the two halves of the gate could disagree about the same metric and
there would be no way to tell which was right.
"""
import re
from dataclasses import asdict, dataclass, field

from .dataset import GoldenItem
from .retrieval import metric_names


def percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile. No numpy dependency and no interpolation surprises
    on the small samples (20-100 questions) this platform actually runs."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 4)
    rank = max(1, min(len(ordered), int(round(pct / 100.0 * len(ordered) + 0.5))))
    return round(ordered[rank - 1], 4)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


@dataclass
class RunSummary:
    total: int = 0
    scored: int = 0
    errors: int = 0

    # Share of questions that came back with any answer text at all. Needs no
    # judge and applies to every item, so unlike the scored metrics it cannot
    # quietly go missing -- which is exactly why it is here. An empty answer is
    # not a refusal and not a wrong answer; without this it was invisible.
    answered_rate: float | None = None
    unanswered: list[str] = field(default_factory=list)

    # {metric: {"measured": n, "applicable": m}} -- see the module docstring.
    coverage: dict[str, dict] = field(default_factory=dict)

    hallucination_rate: float | None = None
    answer_relevance: float | None = None
    faithfulness: float | None = None

    latency_p50: float | None = None
    latency_p95: float | None = None
    cost_per_query: float | None = None
    total_cost_usd: float = 0.0

    input_tokens: int = 0
    output_tokens: int = 0
    judge_calls: int = 0

    # Unanswerable items answered with a refusal, as a fraction. Directly measures
    # "does it admit ignorance instead of inventing" -- the behaviour that matters
    # most in a medical setting and that averaged scores hide.
    abstention_rate: float | None = None
    unanswerable_total: int = 0

    # The other half of that coin. abstention_rate alone is gameable: a prompt that
    # refuses EVERYTHING scores 1.0 and sails through the gate while the app is
    # useless. This is the share of ANSWERABLE items it wrongly refused, and it has
    # to be gated too or the pair is meaningless.
    false_refusal_rate: float | None = None
    answerable_total: int = 0

    # Retrieval, scored against human-labelled gold pages. Keys carry k because a
    # hit rate at 5 and at 20 are different measurements. Populated dynamically so
    # changing k does not need a schema change here.
    retrieval: dict[str, float | None] = field(default_factory=dict)
    retrieval_k: int | None = None
    retrieval_scored: int = 0

    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        # Flatten retrieval into the top level so the gate, the store and the
        # report all look metrics up the same way, by a single name. `coverage`
        # stays nested on purpose: it is keyed BY metric name, so flattening it
        # would collide with the metric values themselves.
        data.update(data.pop("retrieval") or {})
        return data


# How a refusal is recognised. This is configured by the APP, not fixed here:
# the library cannot know how a given app phrases "I don't know", and a
# hard-coded list of English phrases silently made every metric wrong for an app
# that worded it differently or answered in another language.
#
# Two sources, in order of trust:
#
# 1. the target's own `refused` flag -- the app knows whether it took its refusal
#    branch, and no pattern can beat that;
# 2. `refusal.patterns` from llmeval.yaml, as case-insensitive regexes.
#
# Neither available means abstention_rate and false_refusal_rate are UNMEASURABLE
# -- which fails the gate if they are declared, rather than scoring every answer
# as a non-refusal and reporting a perfect 0.000 false-refusal rate.
#
# ONE detector, used by both metrics. If each had its own, the two could disagree
# about whether a given answer was a refusal -- and the pair only means anything
# if both are measuring the same event.


@dataclass(frozen=True)
class RefusalPolicy:
    """How to decide whether one answer was a refusal."""

    use_target_flag: bool = True
    patterns: tuple = ()

    @classmethod
    def from_config(cls, cfg: dict | None) -> "RefusalPolicy":
        block = (cfg or {}).get("refusal") or {}
        return cls(
            use_target_flag=bool(block.get("use_target_flag", True)),
            patterns=tuple(
                re.compile(str(p), re.IGNORECASE) for p in (block.get("patterns") or [])
            ),
        )

    @property
    def can_measure(self) -> bool:
        """Whether this policy can decide anything at all."""
        return bool(self.patterns) or self.use_target_flag

    def verdict(self, record: dict) -> bool | None:
        """True/False, or None when this policy cannot tell."""
        if self.use_target_flag and record.get("refused") is not None:
            return bool(record["refused"])
        if not self.patterns:
            return None
        answer = record.get("answer") or ""
        return any(p.search(answer) for p in self.patterns)


def has_answer(record: dict) -> bool:
    """Whether the target produced any answer text for this question.

    An errored item counts as unanswered too: from the dataset's point of view
    the question went unanswered either way, and both should pull the same
    metric down rather than one of them vanishing from the average.
    """
    if record.get("error"):
        return False
    return bool((record.get("answer") or "").strip())


# Which items each metric APPLIES to. Needed because a None per-item value means
# two different things -- "does not apply here" and "applies but could not be
# measured" -- and only the second is a fault. Without this split, coverage
# cannot be computed and an unmeasured item is indistinguishable from an
# irrelevant one.
def _judgeable(record: dict) -> bool:
    """Whether this item can be scored for GROUNDING: an answer and a context."""
    return has_answer(record) and bool(record.get("contexts"))


def item_metric_applicable(
    results: list[dict],
    retrieval_k: int | None = None,
    refusal: "RefusalPolicy | None" = None,
) -> dict[str, set]:
    names = metric_names(retrieval_k) if retrieval_k else {}
    can_detect_refusal = refusal is not None and refusal.can_measure
    out: dict[str, set] = {
        # Every question should get an answer, including ones that errored.
        "answered_rate": {r["id"] for r in results},
        # Faithfulness checks an answer against the passages it was built from,
        # so it applies only where BOTH exist. A correctly refused question
        # retrieves nothing, and holding faithfulness against it would fail
        # every healthy run on a dataset that contains unanswerable items.
        #
        # This does not reopen the hole it looks like it might: an app that
        # stops retrieving entirely makes faithfulness unmeasurable for every
        # item, and `require_all_metrics` fails the run on the missing
        # run-level value. An app that answers WITHOUT retrieving is caught the
        # same way.
        "faithfulness": {r["id"] for r in results if _judgeable(r)},
        "hallucination_rate": {r["id"] for r in results if _judgeable(r)},
        # Relevance needs only the question and the answer, so it applies
        # wherever there is an answer -- including a refusal, which is exactly
        # how the "fluent but ungrounded" regression was caught: relevance
        # stayed at 0.988 while faithfulness went missing.
        "answer_relevance": {r["id"] for r in results if has_answer(r)},
        # Each half of the refusal pair applies to one kind of question only --
        # and to neither if no refusal policy is configured, since then the
        # question is unmeasurable rather than unanswered.
        "abstention_rate": {
            r["id"]
            for r in results
            if has_answer(r) and r.get("kind") == "unanswerable" and can_detect_refusal
        },
        "false_refusal_rate": {
            r["id"]
            for r in results
            if has_answer(r) and r.get("kind") == "answerable" and can_detect_refusal
        },
    }
    if names:
        # Retrieval is scored against human-labelled gold pages, so it applies
        # only where a label exists. An answerable item with no gold_sources is
        # not a retrieval failure -- it is an unlabelled question.
        scorable = {
            r["id"]
            for r in results
            if r.get("kind") == "answerable" and r.get("gold_sources")
        }
        for key in ("hit", "recall", "rr"):
            out[names[key]] = set(scorable)
    return out


def metric_coverage(
    results: list[dict],
    retrieval_k: int | None = None,
    refusal: "RefusalPolicy | None" = None,
) -> dict[str, dict]:
    """{metric: {"measured": n, "applicable": m}} for every metric."""
    values = item_metric_values(results, retrieval_k, refusal)
    applies = item_metric_applicable(results, retrieval_k, refusal)
    out: dict[str, dict] = {}
    for metric, per_item in values.items():
        relevant = applies.get(metric, set())
        out[metric] = {
            "measured": sum(
                1 for item_id in relevant if per_item.get(item_id) is not None
            ),
            "applicable": len(relevant),
        }
    return out


def looks_like_refusal(answer: str, policy: "RefusalPolicy | None" = None) -> bool:
    """Back-compatible helper. Prefer `RefusalPolicy.verdict`, which can say
    "I cannot tell" -- this collapses that to False."""
    if policy is None:
        return False
    return bool(policy.verdict({"answer": answer}))




def item_metric_values(
    results: list[dict],
    retrieval_k: int | None = None,
    refusal: "RefusalPolicy | None" = None,
) -> dict[str, dict]:
    """Per-metric, per-item values: {metric_name: {item_id: value_or_None}}.

    This is the raw material for both the run summary and the paired comparison,
    produced once so the two can never drift apart.
    """
    names = metric_names(retrieval_k) if retrieval_k else {}
    out: dict[str, dict] = {
        "answered_rate": {},
        "faithfulness": {},
        "answer_relevance": {},
        "hallucination_rate": {},
        "abstention_rate": {},
        "false_refusal_rate": {},
    }
    if names:
        out[names["hit"]] = {}
        out[names["recall"]] = {}
        out[names["rr"]] = {}

    for record in results:
        item_id = record["id"]
        errored = bool(record.get("error"))

        # Did the app say anything at all? Measured here rather than inferred
        # from a missing score, because "the judge could not grade it" and "the
        # app produced nothing" are different failures with different fixes.
        out["answered_rate"][item_id] = 0.0 if not has_answer(record) else 1.0

        out["faithfulness"][item_id] = record.get("faithfulness")
        out["answer_relevance"][item_id] = record.get("answer_relevancy")

        hallucinated = record.get("hallucinated")
        out["hallucination_rate"][item_id] = (
            None if hallucinated is None else (1.0 if hallucinated else 0.0)
        )

        # Each refusal metric applies to exactly one kind of item, and is None on
        # the other -- so the mean of each is over the right denominator.
        # None when the policy cannot tell -- no target flag and no patterns --
        # which keeps "we could not measure this" distinct from "it did not
        # refuse". The gate fails the first and passes the second.
        verdict = None if (errored or refusal is None) else refusal.verdict(record)
        refused = None if verdict is None else (1.0 if verdict else 0.0)
        kind = record.get("kind")
        out["abstention_rate"][item_id] = refused if kind == "unanswerable" else None
        out["false_refusal_rate"][item_id] = refused if kind == "answerable" else None

        if names:
            scores = record.get("retrieval") or {}
            out[names["hit"]][item_id] = scores.get("hit")
            out[names["recall"]][item_id] = scores.get("recall")
            out[names["rr"]][item_id] = scores.get("rr")

    return out


def values_by_item(item_values: dict[str, dict]) -> dict[str, dict]:
    """Transpose {metric: {item_id: value}} into {item_id: {metric: value}}.

    `item_metric_values` is organised by metric because that is what the gate
    and the paired test iterate over. The store needs the other orientation: one
    row per question, with every metric on it. Transposing here keeps a single
    definition of each value -- the store does not recompute anything.
    """
    out: dict[str, dict] = {}
    for metric, per_item in item_values.items():
        for item_id, value in per_item.items():
            out.setdefault(item_id, {})[metric] = value
    return out


def summarize(
    results: list[dict],
    judge_calls: int = 0,
    retrieval_k: int | None = None,
    refusal: "RefusalPolicy | None" = None,
) -> RunSummary:
    """`results` are the per-item dicts produced by the runner."""
    summary = RunSummary(total=len(results), judge_calls=judge_calls, retrieval_k=retrieval_k)
    per_item = item_metric_values(results, retrieval_k, refusal)

    def measured(metric: str) -> list[float]:
        return [v for v in per_item[metric].values() if v is not None]

    summary.errors = sum(1 for r in results if r.get("error"))
    summary.scored = sum(1 for r in results if not r.get("error"))

    summary.coverage = metric_coverage(results, retrieval_k, refusal)
    summary.answered_rate = _mean(measured("answered_rate"))
    summary.unanswered = [r["id"] for r in results if not has_answer(r)]
    if summary.unanswered:
        summary.notes.append(
            f"{len(summary.unanswered)} question(s) produced no answer: "
            f"{', '.join(summary.unanswered)}. An unanswered question cannot be "
            f"graded, so it is EXCLUDED from faithfulness and relevance -- which "
            f"makes those averages look better, not worse. answered_rate and the "
            f"per-metric coverage check are what catch it."
        )

    summary.faithfulness = _mean(measured("faithfulness"))
    summary.answer_relevance = _mean(measured("answer_relevance"))

    # Derived from faithfulness, not from DeepEval's HallucinationMetric -- see
    # the module docstring in metrics.py for why that metric misses additions.
    summary.hallucination_rate = _mean(measured("hallucination_rate"))
    if summary.hallucination_rate is None:
        summary.notes.append("hallucination_rate not measured: no item had faithfulness scored")

    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    summary.latency_p50 = percentile(latencies, 50)
    summary.latency_p95 = percentile(latencies, 95)

    summary.total_cost_usd = round(sum(r.get("cost_usd") or 0.0 for r in results), 6)
    summary.cost_per_query = (
        round(summary.total_cost_usd / len(results), 8) if results else None
    )
    summary.input_tokens = sum(r.get("input_tokens") or 0 for r in results)
    summary.output_tokens = sum(r.get("output_tokens") or 0 for r in results)

    abstention = measured("abstention_rate")
    summary.unanswerable_total = len(abstention)
    summary.abstention_rate = _mean(abstention)
    if not abstention:
        if refusal is None or not refusal.can_measure:
            summary.notes.append(
                "abstention_rate not measured: no way to recognise a refusal. Either "
                "have the target return `refused`, or set refusal.patterns in "
                "llmeval.yaml. Until then this metric and false_refusal_rate cannot "
                "be scored -- which FAILS the gate if they are declared."
            )
        else:
            summary.notes.append(
                "abstention_rate not measured: the dataset slice contains no "
                "unanswerable items. --limit takes the FIRST n items, so mix "
                "unanswerable questions through the file rather than grouping them "
                "at the end."
            )

    false_refusals = measured("false_refusal_rate")
    summary.answerable_total = len(false_refusals)
    summary.false_refusal_rate = _mean(false_refusals)
    if not false_refusals:
        summary.notes.append(
            "false_refusal_rate not measured: the dataset slice contains no answerable items."
        )

    if retrieval_k:
        names = metric_names(retrieval_k)
        summary.retrieval = {
            names["hit"]: _mean(measured(names["hit"])),
            names["recall"]: _mean(measured(names["recall"])),
            names["rr"]: _mean(measured(names["rr"])),
        }
        summary.retrieval_scored = len(measured(names["hit"]))
        if not summary.retrieval_scored:
            summary.notes.append(
                "retrieval metrics not measured: no answerable item had both gold_sources "
                "and a target that reports what it retrieved."
            )

    if summary.total_cost_usd == 0.0:
        summary.notes.append(
            "cost is 0: pricing.* in the app config is unset (free tier). "
            "Fill it in before comparing paid models on cost."
        )
    return summary


def unanswerable_ids(items: list[GoldenItem]) -> set[str]:
    return {item.id for item in items if item.is_unanswerable}
