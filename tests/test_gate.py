"""The gate: floors, the unmeasured rule, paired regressions, and exit codes."""
import pytest

from llm_eval_pipeline.cli import EXIT_GATE_FAILED, EXIT_INCOMPLETE, EXIT_OK, exit_code_for
from llm_eval_pipeline.gate import UnknownMetric, evaluate, parse_specs
from llm_eval_pipeline.paired import PairedVerdict


def verdict(metric: str, regression: bool) -> PairedVerdict:
    return PairedVerdict(
        metric=metric,
        kind="continuous",
        label=metric,
        baseline=0.95,
        current=0.80,
        delta=-0.15,
        paired_n=20,
        regression=regression,
        detail="[-0.20, -0.10]",
        worse_items=["a1", "a2"],
    )


class TestParseSpecs:
    def test_bare_number_is_a_floor(self):
        specs = parse_specs({"faithfulness": 0.9})
        assert specs["faithfulness"].floor == 0.9
        assert specs["faithfulness"].direction == "min"
        assert specs["faithfulness"].kind == "continuous"

    def test_mapping_form(self):
        specs = parse_specs(
            {"false_refusal_rate": {"floor": 0.2, "type": "binary", "direction": "lower_is_better"}}
        )
        spec = specs["false_refusal_rate"]
        assert (spec.floor, spec.kind, spec.direction) == (0.2, "binary", "max")

    def test_at_k_template_resolves(self):
        specs = parse_specs({"retrieval_hit_rate@k": 0.7}, retrieval_k=5)
        assert "retrieval_hit_rate@5" in specs
        assert specs["retrieval_hit_rate@5"].label.endswith("@5")

    def test_explicit_k_also_works(self):
        specs = parse_specs({"retrieval_recall@5": 0.6}, retrieval_k=5)
        assert "retrieval_recall@5" in specs

    def test_unknown_metric_is_rejected(self):
        """A typo must not silently disable a check."""
        with pytest.raises(UnknownMetric):
            parse_specs({"faithfullness": 0.9})

    def test_mapping_without_floor_is_rejected(self):
        with pytest.raises(UnknownMetric):
            parse_specs({"faithfulness": {"type": "continuous"}})

    def test_bad_direction_is_rejected(self):
        with pytest.raises(UnknownMetric):
            parse_specs({"faithfulness": {"floor": 0.9, "direction": "sideways"}})

    def test_latency_is_an_aggregate(self):
        """No per-item pairing exists for p95, so it never gets a paired test."""
        assert parse_specs({"latency_p95": 10.0})["latency_p95"].kind == "aggregate"


class TestFloors:
    def test_min_direction(self):
        assert evaluate({"faithfulness": 0.91}, {"faithfulness": 0.9}).passed
        assert not evaluate({"faithfulness": 0.89}, {"faithfulness": 0.9}).passed

    def test_max_direction(self):
        assert evaluate({"false_refusal_rate": 0.1}, {"false_refusal_rate": 0.2}).passed
        assert not evaluate({"false_refusal_rate": 0.3}, {"false_refusal_rate": 0.2}).passed

    def test_boundary_is_inclusive(self):
        assert evaluate({"faithfulness": 0.9}, {"faithfulness": 0.9}).passed
        assert evaluate({"latency_p95": 10.0}, {"latency_p95": 10.0}).passed


class TestUnmeasuredRule:
    def test_unmeasured_fails_by_default(self):
        """The rule that caught the real regression."""
        result = evaluate({"faithfulness": None}, {"faithfulness": 0.9})
        assert result.passed is False
        assert result.checks[0].actual is None
        assert result.checks[0].note == "not measured"
        assert result.checks[0].failure_reason == "not measured"

    def test_can_be_relaxed_explicitly(self):
        result = evaluate({"faithfulness": None}, {"faithfulness": 0.9}, require_all=False)
        assert result.passed is True
        assert result.checks[0].status == "SKIP"

    def test_relevance_alone_cannot_rescue_a_run(self):
        """Exactly the shape of the regression: relevance high, faithfulness gone."""
        summary = {"answer_relevance": 0.988, "faithfulness": None, "hallucination_rate": None}
        result = evaluate(
            summary,
            {"answer_relevance": 0.9, "faithfulness": 0.9, "hallucination_rate": 0.05},
        )
        assert result.passed is False
        assert {c.metric for c in result.failures} == {"faithfulness", "hallucination_rate"}


class TestPairedIntegration:
    def test_regression_fails_a_metric_inside_its_floor(self):
        """0.80 clears a 0.75 floor, but it dropped 0.15 against the baseline."""
        result = evaluate(
            {"faithfulness": 0.80},
            {"faithfulness": 0.75},
            paired={"faithfulness": verdict("faithfulness", regression=True)},
        )
        assert result.passed is False
        check = result.checks[0]
        assert check.paired_regression is True
        assert check.failure_reason == "regression vs baseline"
        assert check.worse_items == ["a1", "a2"]

    def test_no_regression_passes(self):
        result = evaluate(
            {"faithfulness": 0.80},
            {"faithfulness": 0.75},
            paired={"faithfulness": verdict("faithfulness", regression=False)},
        )
        assert result.passed is True

    def test_floor_still_bites_without_a_paired_result(self):
        result = evaluate({"faithfulness": 0.50}, {"faithfulness": 0.75}, paired=None)
        assert result.passed is False
        assert result.checks[0].failure_reason == "below floor"


class TestExitCodes:
    def test_pass(self):
        assert exit_code_for(evaluate({"faithfulness": 0.95}, {"faithfulness": 0.9})) == EXIT_OK

    def test_gate_failure_blocks(self):
        gate = evaluate({"faithfulness": 0.1}, {"faithfulness": 0.9})
        assert exit_code_for(gate) == EXIT_GATE_FAILED

    def test_hard_failure_does_not_block(self):
        """A quota wall must never read as a quality verdict."""
        gate = evaluate({"faithfulness": 0.1}, {"faithfulness": 0.9})
        gate.hard_failure = "judge out of quota after 3/20 questions"
        assert exit_code_for(gate) == EXIT_INCOMPLETE
