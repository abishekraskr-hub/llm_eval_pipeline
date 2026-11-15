"""Paired comparison: orientation, pairing rules, and when to skip."""
import pytest

from llm_eval_pipeline.gate import parse_specs
from llm_eval_pipeline.paired import PairedConfig, compare


def values(**metrics) -> dict:
    return dict(metrics)


SPECS = parse_specs(
    {
        "faithfulness": 0.75,
        "false_refusal_rate": {"floor": 0.2, "type": "binary", "direction": "lower_is_better"},
        "abstention_rate": {"floor": 0.7, "type": "binary", "direction": "higher_is_better"},
        "latency_p95": 10.0,
    }
)


class TestOrientation:
    def test_higher_is_better_binary(self):
        """abstention: baseline refused, now it doesn't -> worse."""
        base = values(abstention_rate={"u1": 1.0, "u2": 1.0, "u3": 1.0})
        cur = values(abstention_rate={"u1": 0.0, "u2": 0.0, "u3": 1.0})
        result = compare(cur, base, SPECS, PairedConfig(min_flips=2))
        got = result.verdicts["abstention_rate"]
        assert (got.mcnemar.worse, got.mcnemar.better) == (2, 0)
        assert got.worse_items == ["u1", "u2"]

    def test_lower_is_better_binary_inverts(self):
        """false refusal: baseline answered (0 = good), now refuses (1 = bad).

        Six items, not three: three unanimous flips give p = 0.25, so a smaller
        example would be testing the orientation and the significance threshold at
        the same time and failing for the wrong reason.
        """
        ids = [f"a{i}" for i in range(6)]
        base = values(false_refusal_rate={i: 0.0 for i in ids})
        cur = values(false_refusal_rate={i: 1.0 for i in ids})
        result = compare(cur, base, SPECS, PairedConfig(min_flips=2))
        got = result.verdicts["false_refusal_rate"]
        assert (got.mcnemar.worse, got.mcnemar.better) == (6, 0)
        assert got.regression is True

    def test_three_flips_is_not_enough_to_block(self):
        """Orientation is right, but the evidence is too thin -- by design."""
        base = values(false_refusal_rate={"a1": 0.0, "a2": 0.0, "a3": 0.0})
        cur = values(false_refusal_rate={"a1": 1.0, "a2": 1.0, "a3": 1.0})
        got = compare(cur, base, SPECS, PairedConfig()).verdicts["false_refusal_rate"]
        assert (got.mcnemar.worse, got.mcnemar.better) == (3, 0)
        assert got.regression is False

    def test_lower_is_better_improvement_is_not_a_regression(self):
        base = values(false_refusal_rate={f"a{i}": 1.0 for i in range(6)})
        cur = values(false_refusal_rate={f"a{i}": 0.0 for i in range(6)})
        result = compare(cur, base, SPECS, PairedConfig())
        assert result.verdicts["false_refusal_rate"].regression is False

    def test_continuous_higher_is_better(self):
        base = values(faithfulness={f"a{i}": 1.0 for i in range(20)})
        cur = values(faithfulness={f"a{i}": 0.8 for i in range(20)})
        result = compare(cur, base, SPECS, PairedConfig(bootstrap_resamples=1000))
        got = result.verdicts["faithfulness"]
        assert got.bootstrap.mean_delta == pytest.approx(-0.2)
        assert got.regression is True
        assert len(got.worse_items) == 20

    def test_continuous_improvement_passes(self):
        base = values(faithfulness={f"a{i}": 0.6 for i in range(20)})
        cur = values(faithfulness={f"a{i}": 0.9 for i in range(20)})
        result = compare(cur, base, SPECS, PairedConfig(bootstrap_resamples=1000))
        assert result.verdicts["faithfulness"].regression is False
        assert result.verdicts["faithfulness"].worse_items == []


class TestPairing:
    def test_items_only_in_one_run_are_ignored_and_counted(self):
        base = values(faithfulness={"a1": 1.0, "a2": 1.0, "gone": 1.0})
        cur = values(faithfulness={"a1": 1.0, "a2": 1.0, "new": 1.0})
        result = compare(cur, base, SPECS, PairedConfig(min_pair_coverage=0.5))
        assert result.paired_items == 2
        assert result.unpaired_current == 1
        assert result.unpaired_baseline == 1
        assert result.verdicts["faithfulness"].paired_n == 2

    def test_low_coverage_skips_with_a_reason(self):
        base = values(faithfulness={"a1": 1.0})
        cur = values(faithfulness={f"a{i}": 1.0 for i in range(10)})
        result = compare(cur, base, SPECS, PairedConfig(min_pair_coverage=0.8))
        assert result.skipped_reason is not None
        assert "min_pair_coverage" in result.skipped_reason
        assert result.verdicts == {}

    def test_items_unmeasured_in_either_run_are_not_paired(self):
        """None on either side means there is no pair, not a zero."""
        base = values(faithfulness={"a1": 1.0, "a2": None})
        cur = values(faithfulness={"a1": 0.5, "a2": 0.5})
        result = compare(cur, base, SPECS, PairedConfig(bootstrap_resamples=500))
        assert result.verdicts["faithfulness"].paired_n == 1

    def test_metric_with_no_pairs_is_reported_not_crashed(self):
        base = values(faithfulness={"a1": None})
        cur = values(faithfulness={"a1": None})
        result = compare(cur, base, SPECS, PairedConfig())
        got = result.verdicts["faithfulness"]
        assert got.paired_n == 0 and got.regression is False
        assert "both runs" in got.note


class TestScope:
    def test_aggregate_metrics_get_no_paired_test(self):
        """p95 is a property of the run, not of a question."""
        base = values(latency_p95={"a1": 1.0})
        cur = values(latency_p95={"a1": 99.0})
        result = compare(cur, base, SPECS, PairedConfig(min_pair_coverage=0.0))
        assert "latency_p95" not in result.verdicts

    def test_empty_current_run_skips(self):
        result = compare({}, values(faithfulness={"a1": 1.0}), SPECS, PairedConfig())
        assert result.skipped_reason == "no items in this run"


class TestConfig:
    def test_defaults_match_the_spec(self):
        cfg = PairedConfig()
        assert (cfg.alpha, cfg.min_flips, cfg.min_delta) == (0.05, 2, 0.03)
        assert (cfg.bootstrap_resamples, cfg.seed) == (10000, 42)
        assert cfg.min_pair_coverage == 0.8
        # The integration branch. Nothing is pushed to main, so a run there could
        # never be a baseline. See TestBaselineBranch in test_storage.py.
        assert cfg.base_branch == "develop"

    def test_unknown_key_is_rejected(self):
        """A typo in eval_config.yaml must not silently disable the gate."""
        with pytest.raises(ValueError, match="unknown paired config key"):
            PairedConfig.from_dict({"alpah": 0.1})

    def test_from_dict_overrides(self):
        cfg = PairedConfig.from_dict({"alpha": 0.01, "enabled": False})
        assert cfg.alpha == 0.01 and cfg.enabled is False
