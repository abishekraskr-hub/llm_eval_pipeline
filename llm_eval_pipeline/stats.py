"""Statistical tests for the paired gate. Standard library only.

Why a statistical gate at all. On a 20-100 item benchmark, scores move between
runs with no code change: judges are stochastic, and one item flipping is worth
5% on a 20-item set. A fixed threshold at 0.90 will therefore block PRs that
changed nothing, and a team that sees the gate cry wolf twice stops reading it.

The fix is to stop asking "is today's number above the line?" and start asking
"did THESE questions get worse than they were on the baseline run?" -- which is a
paired comparison, and paired comparisons are far more sensitive, because each
question acts as its own control.

Two tests, because there are two kinds of metric:

* per-item 0/1 outcomes (did it refuse, did retrieval hit) -> exact McNemar
* per-item scores (faithfulness, recall) -> paired bootstrap of the mean delta

Both are deterministic: the binomial is exact rather than sampled, and the
bootstrap is seeded. A gate that returned a different verdict on a re-run would be
worse than no gate.
"""
import math
import random
import statistics
from dataclasses import dataclass


def exact_binomial_two_sided(k: int, n: int) -> float:
    """Two-sided exact binomial p-value for k of n, under p = 0.5.

    Exact, not normal-approximated: n here is the number of items that CHANGED
    between the two runs, which is routinely under 10, and the normal
    approximation is poor there -- it would overstate significance and block PRs
    on three coin flips.

    Because p = 0.5 the distribution is symmetric, so the two-sided value is just
    twice the larger tail, capped at 1.
    """
    if n <= 0:
        return 1.0
    far = max(k, n - k)
    tail = sum(math.comb(n, i) for i in range(far, n + 1))
    return min(1.0, 2.0 * tail / (2**n))


@dataclass
class McNemarResult:
    worse: int
    better: int
    n: int  # items that changed; unchanged items carry no information
    p_value: float
    regression: bool
    note: str = ""

    @property
    def summary(self) -> str:
        if self.n == 0:
            return "no items changed"
        return f"{self.worse} worse / {self.better} better, p = {self.p_value:.3f}"


def mcnemar(
    worse: int,
    better: int,
    alpha: float = 0.05,
    min_flips: int = 2,
) -> McNemarResult:
    """Exact McNemar test on paired binary outcomes.

    `worse`  = good on the baseline, bad now
    `better` = bad on the baseline, good now

    Items that did not change are deliberately excluded: McNemar's whole premise
    is that only discordant pairs carry information about a change.

    `min_flips` is an effect-size floor on top of significance. With a large
    dataset a 2-vs-0 flip can clear p < 0.05 while being clinically irrelevant;
    requiring a minimum net movement keeps the gate about regressions that matter
    rather than regressions that are merely detectable.
    """
    n = worse + better
    p_value = exact_binomial_two_sided(worse, n)
    regression = worse > better and p_value < alpha and (worse - better) >= min_flips
    note = ""
    if worse > better and p_value < alpha and (worse - better) < min_flips:
        note = f"significant but below min_flips={min_flips}"
    return McNemarResult(
        worse=worse, better=better, n=n, p_value=p_value, regression=regression, note=note
    )


def _percentile(ordered: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not ordered:
        raise ValueError("empty sample")
    rank = max(1, min(len(ordered), int(round(pct / 100.0 * len(ordered) + 0.5))))
    return ordered[rank - 1]


@dataclass
class BootstrapResult:
    n: int
    mean_delta: float
    ci_low: float
    ci_high: float
    sem: float | None
    regression: bool
    note: str = ""

    @property
    def summary(self) -> str:
        return f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}]"


def paired_bootstrap(
    deltas: list[float],
    alpha: float = 0.05,
    min_delta: float = 0.03,
    resamples: int = 10000,
    seed: int = 42,
) -> BootstrapResult:
    """Bootstrap a confidence interval for the mean paired difference.

    `deltas` must already be oriented so that **negative means worse**; the caller
    flips the sign for lower-is-better metrics. A regression needs both:

    * the whole confidence interval below zero -- we are confident it dropped, and
    * |mean| >= min_delta -- the drop is big enough to care about.

    Bootstrapping rather than a t-test because these per-item scores are bounded in
    [0, 1] and often piled up at 1.0, which is nothing like normal. Resampling
    makes no distributional assumption.
    """
    n = len(deltas)
    if n == 0:
        return BootstrapResult(0, 0.0, 0.0, 0.0, None, False, "no paired items")

    mean_delta = statistics.fmean(deltas)
    sem = statistics.stdev(deltas) / math.sqrt(n) if n > 1 else None

    if n == 1:
        # One pair carries no interval. Treat it as inconclusive rather than
        # inventing a zero-width CI that could fail a PR on a single question.
        return BootstrapResult(
            1, mean_delta, mean_delta, mean_delta, None, False, "only 1 paired item"
        )

    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = rng.choices(deltas, k=n)
        means.append(statistics.fmean(sample))
    means.sort()

    ci_low = _percentile(means, 100.0 * (alpha / 2.0))
    ci_high = _percentile(means, 100.0 * (1.0 - alpha / 2.0))

    regression = ci_high < 0.0 and abs(mean_delta) >= min_delta
    note = ""
    if ci_high < 0.0 and abs(mean_delta) < min_delta:
        note = f"consistent drop but below min_delta={min_delta}"
    return BootstrapResult(
        n=n,
        mean_delta=mean_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        sem=sem,
        regression=regression,
        note=note,
    )
