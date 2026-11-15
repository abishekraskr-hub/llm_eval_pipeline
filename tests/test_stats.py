"""The statistical tests, including every case the spec pins down by number."""
import random

import pytest

from llm_eval_pipeline.stats import exact_binomial_two_sided, mcnemar, paired_bootstrap


class TestExactBinomial:
    def test_spec_values(self):
        assert exact_binomial_two_sided(9, 10) == pytest.approx(0.0215, abs=1e-4)
        assert exact_binomial_two_sided(6, 10) == pytest.approx(0.754, abs=1e-3)

    def test_symmetric(self):
        """p = 0.5, so k and n-k are indistinguishable."""
        for k in range(11):
            assert exact_binomial_two_sided(k, 10) == exact_binomial_two_sided(10 - k, 10)

    def test_degenerate(self):
        assert exact_binomial_two_sided(0, 0) == 1.0
        assert exact_binomial_two_sided(5, 10) == 1.0  # dead even

    def test_never_exceeds_one(self):
        for n in range(0, 25):
            for k in range(n + 1):
                assert 0.0 <= exact_binomial_two_sided(k, n) <= 1.0


class TestMcNemar:
    def test_nine_worse_one_better_is_a_regression(self):
        result = mcnemar(worse=9, better=1)
        assert result.p_value == pytest.approx(0.0215, abs=1e-4)
        assert result.regression is True

    def test_six_worse_four_better_is_noise(self):
        result = mcnemar(worse=6, better=4)
        assert result.p_value == pytest.approx(0.754, abs=1e-3)
        assert result.regression is False

    def test_no_change_passes(self):
        result = mcnemar(worse=0, better=0)
        assert result.n == 0
        assert result.p_value == 1.0
        assert result.regression is False

    def test_real_but_too_small_passes(self):
        """worse=2, better=0 with min_flips=3: a true drop, below the effect floor."""
        result = mcnemar(worse=2, better=0, min_flips=3)
        assert result.regression is False

    def test_min_flips_note_explains_a_near_miss(self):
        result = mcnemar(worse=6, better=0, min_flips=9)
        assert result.p_value < 0.05  # significant
        assert result.regression is False  # but below the effect floor
        assert "min_flips" in result.note

    def test_six_unanimous_flips_is_the_minimum_detectable_regression(self):
        """A property worth knowing before trusting the gate on a small set.

        With p = 0.5 two-sided, five items all moving the same way gives p = 0.0625
        -- NOT significant. Six gives 0.031. So a 20-question PR run cannot flag a
        binary regression unless at least 6 questions flip, and no threshold
        tuning changes that; only a bigger dataset does.
        """
        assert mcnemar(worse=5, better=0).p_value == pytest.approx(0.0625)
        assert mcnemar(worse=5, better=0).regression is False
        assert mcnemar(worse=6, better=0).p_value == pytest.approx(0.03125)
        assert mcnemar(worse=6, better=0).regression is True

    def test_improvement_is_never_a_regression(self):
        assert mcnemar(worse=1, better=9).regression is False

    def test_unchanged_items_are_excluded(self):
        """n counts only discordant pairs -- that is McNemar's whole premise."""
        assert mcnemar(worse=3, better=2).n == 5


class TestPairedBootstrap:
    def test_uniform_drop_is_a_regression(self):
        result = paired_bootstrap([-0.10] * 30, resamples=2000)
        assert result.mean_delta == pytest.approx(-0.10)
        assert result.ci_high < 0
        assert result.regression is True

    def test_symmetric_noise_passes(self):
        """Mirrored pairs, so the mean is exactly 0 by construction.

        Drawing from a seeded RNG is not good enough here: a 40-sample uniform draw
        can easily land a mean of -0.04 with a CI entirely below zero, which would
        make this test assert the opposite of what it means to.
        """
        deltas = [0.15, -0.15] * 20
        result = paired_bootstrap(deltas, resamples=2000)
        assert result.mean_delta == pytest.approx(0.0)
        assert result.ci_low < 0 < result.ci_high
        assert result.regression is False

    def test_noisy_but_centred_data_passes(self):
        rng = random.Random(7)
        half = [rng.uniform(0.0, 0.2) for _ in range(20)]
        deltas = half + [-d for d in half]  # forced symmetry
        result = paired_bootstrap(deltas, resamples=2000)
        assert result.regression is False

    def test_consistent_but_tiny_drop_passes(self):
        """Mean drop 0.01 with a tight CI, min_delta 0.03: real but not material."""
        result = paired_bootstrap([-0.01] * 30, min_delta=0.03, resamples=2000)
        assert result.ci_high < 0  # confident it dropped
        assert result.regression is False  # but below min_delta
        assert "min_delta" in result.note

    def test_improvement_is_never_a_regression(self):
        result = paired_bootstrap([0.2] * 20, resamples=2000)
        assert result.mean_delta > 0
        assert result.regression is False

    def test_deterministic_for_a_given_seed(self):
        rng = random.Random(3)
        deltas = [rng.uniform(-0.5, 0.3) for _ in range(25)]
        a = paired_bootstrap(deltas, resamples=1500, seed=42)
        b = paired_bootstrap(deltas, resamples=1500, seed=42)
        assert (a.ci_low, a.ci_high, a.regression) == (b.ci_low, b.ci_high, b.regression)

    def test_single_pair_is_inconclusive(self):
        """One question must never be enough to block a PR."""
        result = paired_bootstrap([-0.9])
        assert result.regression is False
        assert "1 paired item" in result.note

    def test_no_pairs(self):
        result = paired_bootstrap([])
        assert result.n == 0 and result.regression is False

    def test_sem_reported(self):
        result = paired_bootstrap([-0.1, -0.2, -0.3, -0.4], resamples=500)
        assert result.sem is not None and result.sem > 0
