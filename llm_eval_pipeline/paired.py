"""Paired comparison of this run against the last good baseline run.

The question this answers is not "is the number above the line?" but "did these
same questions get worse than they were?" -- which is a much sharper instrument on
a small benchmark, because each question acts as its own control.

Comparability is enforced, not assumed. A baseline is only usable if it asked the
same questions (`dataset_hash`) and was graded by the same judge
(`judge_config_hash`). Comparing across either is meaningless: a faithfulness of
0.91 from one judge is not the same measurement as 0.91 from another, and a
"regression" found that way would be an artefact.

When no comparable baseline exists -- the first ever run, a dataset edit, a judge
change -- the paired test is SKIPPED, not failed. The absolute floors carry the
gate, and the report says so plainly.
"""
import statistics
from dataclasses import dataclass, field

from .gate import MetricSpec
from .stats import BootstrapResult, McNemarResult, mcnemar, paired_bootstrap


@dataclass
class PairedConfig:
    enabled: bool = True
    # The integration branch whose last passing run is the baseline.
    base_branch: str = "develop"
    alpha: float = 0.05
    # Binary: net flips (worse - better) must reach this, so a statistically
    # detectable two-item change does not block a PR on its own.
    min_flips: int = 2
    # Continuous: the mean drop must reach this to count as a regression.
    min_delta: float = 0.03
    bootstrap_resamples: int = 10000
    seed: int = 42
    # If fewer than this fraction of current items exist in the baseline, the runs
    # are too different to pair and the test is skipped.
    min_pair_coverage: float = 0.8

    @classmethod
    def from_dict(cls, data: dict | None) -> "PairedConfig":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"unknown paired config key(s): {sorted(unknown)}; known: {sorted(known)}"
            )
        return cls(**data)


@dataclass
class PairedVerdict:
    metric: str
    kind: str
    label: str
    baseline: float | None
    current: float | None
    delta: float | None
    paired_n: int
    regression: bool
    detail: str  # "p = 0.039" or "[-0.040, +0.020]"
    worse_items: list[str] = field(default_factory=list)
    note: str = ""
    sem_current: float | None = None
    sem_baseline: float | None = None
    mcnemar: McNemarResult | None = None
    bootstrap: BootstrapResult | None = None


@dataclass
class PairedComparison:
    baseline_run_id: int | None = None
    baseline_label: str | None = None
    verdicts: dict[str, PairedVerdict] = field(default_factory=dict)
    skipped_reason: str | None = None
    paired_items: int = 0
    unpaired_current: int = 0
    unpaired_baseline: int = 0

    @property
    def ran(self) -> bool:
        return self.skipped_reason is None and bool(self.verdicts)

    @property
    def regressions(self) -> list[PairedVerdict]:
        return [v for v in self.verdicts.values() if v.regression]


def _good(value: float, direction: str) -> int:
    """Orient a binary per-item value so 1 = good.

    `direction == "min"` means higher is better (a hit, a correct refusal), so the
    value passes through. `"max"` means lower is better (a false refusal, a
    hallucination), so it inverts.
    """
    raw = 1 if value >= 0.5 else 0
    return raw if direction == "min" else 1 - raw


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sem(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values) / (len(values) ** 0.5)


def _all_item_ids(values: dict[str, dict]) -> set[str]:
    ids: set[str] = set()
    for per_item in values.values():
        ids.update(per_item)
    return ids


def compare(
    current_values: dict[str, dict],
    baseline_values: dict[str, dict],
    specs: dict[str, MetricSpec],
    cfg: PairedConfig,
    baseline_run_id: int | None = None,
    baseline_label: str | None = None,
) -> PairedComparison:
    """Run the paired tests. `*_values` are {metric: {item_id: value_or_None}}."""
    result = PairedComparison(baseline_run_id=baseline_run_id, baseline_label=baseline_label)

    current_ids = _all_item_ids(current_values)
    baseline_ids = _all_item_ids(baseline_values)
    shared = current_ids & baseline_ids
    result.paired_items = len(shared)
    result.unpaired_current = len(current_ids - baseline_ids)
    result.unpaired_baseline = len(baseline_ids - current_ids)

    if not current_ids:
        result.skipped_reason = "no items in this run"
        return result

    coverage = len(shared) / len(current_ids)
    if coverage < cfg.min_pair_coverage:
        result.skipped_reason = (
            f"only {len(shared)}/{len(current_ids)} items ({coverage:.0%}) exist in the "
            f"baseline, below min_pair_coverage={cfg.min_pair_coverage:.0%}"
        )
        return result

    for name, spec in specs.items():
        if spec.kind == "aggregate":
            # p95 and cost are run-level; there is no per-item pair to form.
            continue
        cur_items = current_values.get(name) or {}
        base_items = baseline_values.get(name) or {}

        # Only items measured in BOTH runs can be paired.
        pairs = [
            (item_id, base_items[item_id], cur_items[item_id])
            for item_id in sorted(shared)
            if cur_items.get(item_id) is not None and base_items.get(item_id) is not None
        ]
        cur_vals = [c for _, _, c in pairs]
        base_vals = [b for _, b, _ in pairs]

        if not pairs:
            result.verdicts[name] = PairedVerdict(
                metric=name,
                kind=spec.kind,
                label=spec.label,
                baseline=None,
                current=None,
                delta=None,
                paired_n=0,
                regression=False,
                detail="-",
                note="no items measured in both runs",
            )
            continue

        cur_mean, base_mean = _mean(cur_vals), _mean(base_vals)

        if spec.kind == "binary":
            worse_items = [
                item_id
                for item_id, b, c in pairs
                if _good(b, spec.direction) == 1 and _good(c, spec.direction) == 0
            ]
            better = sum(
                1
                for _, b, c in pairs
                if _good(b, spec.direction) == 0 and _good(c, spec.direction) == 1
            )
            test = mcnemar(
                worse=len(worse_items),
                better=better,
                alpha=cfg.alpha,
                min_flips=cfg.min_flips,
            )
            result.verdicts[name] = PairedVerdict(
                metric=name,
                kind=spec.kind,
                label=spec.label,
                baseline=base_mean,
                current=cur_mean,
                delta=(cur_mean - base_mean) if None not in (cur_mean, base_mean) else None,
                paired_n=len(pairs),
                regression=test.regression,
                detail=test.summary,
                worse_items=worse_items,
                note=test.note,
                mcnemar=test,
            )
            continue

        # Continuous: orient so negative == worse, whichever way the metric runs.
        deltas = [
            (c - b) if spec.direction == "min" else (b - c)
            for _, b, c in pairs
        ]
        boot = paired_bootstrap(
            deltas,
            alpha=cfg.alpha,
            min_delta=cfg.min_delta,
            resamples=cfg.bootstrap_resamples,
            seed=cfg.seed,
        )
        # Worst drop first, so the report lists the items most worth opening.
        delta_by_item = {item_id: d for (item_id, _, _), d in zip(pairs, deltas)}
        worse_items = sorted(
            (item_id for item_id, d in delta_by_item.items() if d < 0),
            key=lambda item_id: delta_by_item[item_id],
        )
        result.verdicts[name] = PairedVerdict(
            metric=name,
            kind=spec.kind,
            label=spec.label,
            baseline=base_mean,
            current=cur_mean,
            delta=(cur_mean - base_mean) if None not in (cur_mean, base_mean) else None,
            paired_n=len(pairs),
            regression=boot.regression,
            detail=boot.summary,
            worse_items=worse_items,
            note=boot.note,
            sem_current=_sem(cur_vals),
            sem_baseline=_sem(base_vals),
            bootstrap=boot,
        )

    return result
