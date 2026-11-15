"""Report rendering: the bits a PR reader actually looks at."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_eval_pipeline.gate import evaluate
from llm_eval_pipeline.report import console_report, markdown_report
from llm_eval_pipeline.paired import PairedComparison


def test_retrieval_metadata_is_not_rendered_as_a_metric():
    """`retrieval_k: 5` must not appear in the metric table as "5.000"."""
    summary = {
        "retrieval_hit_rate@5": 0.9,
        "retrieval_k": 5,
        "retrieval_scored": 8,
        "notes": [],
    }
    gate = evaluate(summary, {"retrieval_hit_rate@5": 0.7})
    md = markdown_report(summary, gate, [], {}, "stub:judge")
    assert "| retrieval_hit_rate@5 | 0.900 |" in md
    assert "| retrieval_k |" not in md
    assert "5.000" not in md


def test_hard_failure_report_gives_no_quality_verdict():
    gate = evaluate({"faithfulness": 0.1}, {"faithfulness": 0.9})
    gate.hard_failure = "judge out of quota after 3/20 questions"
    md = markdown_report({}, gate, [], {}, "stub:judge")
    assert "DID NOT COMPLETE" in md
    assert "not a quality failure" in md
    # It must not print a threshold table that looks like a verdict.
    assert "FAILED" not in md.splitlines()[0]


def test_console_report_flags_a_paired_regression():
    summary = {"faithfulness": 0.8, "notes": []}
    gate = evaluate({"faithfulness": 0.8}, {"faithfulness": 0.75})
    gate.checks[0].paired_regression = True
    gate.checks[0].passed = False
    gate.passed = False
    out = console_report(summary, gate, PairedComparison(skipped_reason=None))
    assert "regression vs baseline" in out
    assert "GATE: FAILED" in out


def test_skipped_paired_test_is_stated_plainly():
    summary = {"faithfulness": 0.9, "notes": []}
    gate = evaluate(summary, {"faithfulness": 0.75})
    paired = PairedComparison(skipped_reason="no comparable baseline run")
    md = markdown_report(summary, gate, [], {}, "stub:judge", paired=paired)
    assert "No comparable baseline" in md
    assert "absolute floors only" in md
