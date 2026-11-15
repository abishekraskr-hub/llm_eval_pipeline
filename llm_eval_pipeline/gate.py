"""The quality gate: compare a run against thresholds AND against a baseline run.

Each threshold declares its direction explicitly ("max" or "min") rather than
inferring it from the metric name. Getting a direction wrong silently inverts the
gate -- it would pass exactly the runs it exists to block -- so it is stated as
data, not guessed.

A metric FAILS for any of four independent reasons:

1. it was declared but could not be measured at all,
2. it was measured on only PART of the items it applies to,
3. the paired comparison against the baseline detects a real regression, or
4. it breaks its absolute floor.

(4) is a safety floor, set well below normal, that catches disasters. (3) does the
day-to-day work, because on a small benchmark a fixed threshold either flaps or
sits so low it catches nothing. (1) is non-negotiable: a metric that quietly stops
being measured must never read as a pass -- that rule is the only thing that
caught the regression where retrieval was deleted and answer relevance stayed at
0.988.

(2) exists because (1) turned out to be too coarse. It asks "was this measured at
ALL", so one item measuring was enough to satisfy it. A real run then returned an
empty answer for one question in three; that item was dropped from the mean
rather than counted against it, and the gate reported faithfulness 1.000 and
passed every quality check. Partial coverage now fails, because an average over
the items that happened to work is not a measurement of the dataset.
"""
import re
from dataclasses import dataclass, field

# metric name -> ("max"|"min", human label)
#   "min" = the threshold is a MINIMUM, higher is better
#   "max" = the threshold is a MAXIMUM, lower is better
# `@k` entries are templates: a config key of retrieval_hit_rate@k and a resolved
# summary key of retrieval_hit_rate@5 both find this row.
DIRECTIONS: dict[str, tuple[str, str]] = {
    # Share of questions that got any answer at all. Needs no judge, so it is the
    # one quality signal that cannot itself go missing.
    "answered_rate": ("min", "Answered"),
    "hallucination_rate": ("max", "Hallucination rate"),
    "answer_relevance": ("min", "Answer relevance"),
    "faithfulness": ("min", "Faithfulness"),
    "latency_p50": ("max", "p50 latency (s)"),
    "latency_p95": ("max", "p95 latency (s)"),
    "cost_per_query": ("max", "Cost per query (USD)"),
    "abstention_rate": ("min", "Abstention on unanswerable"),
    "false_refusal_rate": ("max", "False refusal on answerable"),
    "retrieval_hit_rate@k": ("min", "Retrieval hit rate"),
    "retrieval_recall@k": ("min", "Retrieval recall"),
    "retrieval_mrr": ("min", "Retrieval MRR"),
}

# Which paired test a metric is eligible for. "aggregate" metrics have no
# meaningful per-item pairing: p95 is a property of the whole run, not of any one
# question, so it stays on its absolute gate alone.
KINDS: dict[str, str] = {
    "answered_rate": "binary",
    "hallucination_rate": "binary",
    "abstention_rate": "binary",
    "false_refusal_rate": "binary",
    "retrieval_hit_rate@k": "binary",
    "faithfulness": "continuous",
    "answer_relevance": "continuous",
    "retrieval_recall@k": "continuous",
    "retrieval_mrr": "continuous",
    "latency_p50": "aggregate",
    "latency_p95": "aggregate",
    "cost_per_query": "aggregate",
}

_AT_K = re.compile(r"@\d+$")

_DIRECTION_ALIASES = {
    "higher_is_better": "min",
    "lower_is_better": "max",
    "min": "min",
    "max": "max",
}


def canonical(metric: str) -> str:
    """Map a resolved metric name back to its template key (`@5` -> `@k`)."""
    return _AT_K.sub("@k", metric)


class UnknownMetric(ValueError):
    pass


@dataclass
class MetricSpec:
    """How one metric is gated."""

    name: str  # resolved, e.g. retrieval_hit_rate@5
    direction: str  # "min" | "max"
    label: str
    kind: str  # "binary" | "continuous" | "aggregate"
    floor: float


def parse_specs(thresholds: dict, retrieval_k: int | None = None) -> dict[str, MetricSpec]:
    """Read the `thresholds:` block, in either supported form.

        faithfulness: 0.90                      # bare number: the floor
        faithfulness:                           # mapping: floor + overrides
          floor: 0.75
          type: continuous
          direction: higher_is_better

    The bare form is kept because it is what the config already used and it stays
    readable for the common case. The mapping form exists so a metric can override
    its type or direction without a code change.
    """
    specs: dict[str, MetricSpec] = {}
    for raw_name, value in (thresholds or {}).items():
        name = raw_name.replace("@k", f"@{retrieval_k}") if retrieval_k else raw_name
        key = canonical(name)
        if key not in DIRECTIONS:
            raise UnknownMetric(
                f"no direction defined for threshold {raw_name!r}; "
                f"known metrics: {sorted(DIRECTIONS)}"
            )
        direction, label = DIRECTIONS[key]
        kind = KINDS.get(key, "aggregate")

        if isinstance(value, dict):
            if "floor" not in value:
                raise UnknownMetric(f"threshold {raw_name!r} is a mapping but has no 'floor'")
            floor = float(value["floor"])
            if "direction" in value:
                alias = str(value["direction"])
                if alias not in _DIRECTION_ALIASES:
                    raise UnknownMetric(
                        f"threshold {raw_name!r}: direction must be one of "
                        f"{sorted(_DIRECTION_ALIASES)}, got {alias!r}"
                    )
                direction = _DIRECTION_ALIASES[alias]
            if "type" in value:
                kind = str(value["type"])
                if kind not in ("binary", "continuous", "aggregate"):
                    raise UnknownMetric(
                        f"threshold {raw_name!r}: type must be binary|continuous|aggregate, "
                        f"got {kind!r}"
                    )
        else:
            floor = float(value)

        label_with_k = f"{label}@{retrieval_k}" if retrieval_k and _AT_K.search(name) else label
        specs[name] = MetricSpec(
            name=name, direction=direction, label=label_with_k, kind=kind, floor=floor
        )
    return specs


@dataclass
class Check:
    metric: str
    label: str
    direction: str
    threshold: float
    actual: float | None
    passed: bool
    skipped: bool = False
    note: str = ""
    # Paired-comparison outcome for this metric, when one ran.
    paired_regression: bool = False
    paired_detail: str = ""
    worse_items: list[str] = field(default_factory=list)
    # How many of the items this metric applies to actually got a value.
    measured: int | None = None
    applicable: int | None = None
    partial_coverage: bool = False

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        return "PASS" if self.passed else "FAIL"

    @property
    def coverage(self) -> float | None:
        """Fraction of applicable items that got a value. None when unknown."""
        if not self.applicable:
            return None
        return (self.measured or 0) / self.applicable

    @property
    def failure_reason(self) -> str:
        if self.passed:
            return ""
        if self.actual is None:
            return "not measured"
        if self.partial_coverage:
            return f"measured on only {self.measured}/{self.applicable} items"
        if self.paired_regression:
            return "regression vs baseline"
        return "below floor" if self.direction == "min" else "above ceiling"


@dataclass
class GateResult:
    checks: list[Check]
    passed: bool
    hard_failure: str | None = None  # run-level problem, e.g. quota

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed and not c.skipped]


def evaluate(
    summary_dict: dict,
    thresholds: dict,
    require_all: bool = True,
    paired: dict | None = None,
    retrieval_k: int | None = None,
    min_metric_coverage: float = 0.9,
) -> GateResult:
    """`require_all=True` fails the gate when a threshold's metric was never measured.

    That default is deliberate: a metric silently going missing (judge errored, app
    returned no context) must not look like a pass. CI would otherwise go green on
    a run that measured nothing.

    `min_metric_coverage` is the other half of that rule, and the sharper one. A
    metric must be measured on at least this fraction of the items it APPLIES to,
    or it fails -- because `None` values are excluded from the mean, so an item
    the judge could not score makes the average look better rather than worse.
    Set it to 0 to switch the check off; 1.0 demands every applicable item.

    `paired` maps a resolved metric name to a PairedVerdict. Absent or None means
    no comparable baseline was found, which is not a failure -- the floors simply
    carry the whole gate for that run.
    """
    specs = parse_specs(thresholds, retrieval_k)
    coverage = summary_dict.get("coverage") or {}
    checks: list[Check] = []

    for name, spec in specs.items():
        actual = summary_dict.get(name)
        verdict = (paired or {}).get(name)
        counts = coverage.get(name) or {}
        measured = counts.get("measured")
        applicable = counts.get("applicable")

        if actual is None:
            checks.append(
                Check(
                    metric=name,
                    label=spec.label,
                    direction=spec.direction,
                    threshold=spec.floor,
                    actual=None,
                    passed=not require_all,
                    skipped=not require_all,
                    note="not measured",
                )
            )
            continue

        within_floor = (
            actual <= spec.floor if spec.direction == "max" else actual >= spec.floor
        )
        regressed = bool(verdict and verdict.regression)
        # An aggregate metric (p95, cost) has no per-item values, so coverage
        # does not apply to it and is not held against it.
        partial = bool(
            min_metric_coverage > 0
            and spec.kind != "aggregate"
            and applicable
            and (measured or 0) / applicable < min_metric_coverage
        )
        checks.append(
            Check(
                metric=name,
                label=spec.label,
                direction=spec.direction,
                threshold=spec.floor,
                actual=float(actual),
                passed=within_floor and not regressed and not partial,
                measured=measured,
                applicable=applicable,
                partial_coverage=partial,
                paired_regression=regressed,
                paired_detail=verdict.detail if verdict else "",
                worse_items=list(verdict.worse_items) if verdict else [],
                note=(verdict.note if verdict and verdict.note else ""),
            )
        )

    return GateResult(checks=checks, passed=all(c.passed for c in checks))
