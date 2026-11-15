"""Whole-pipeline tests with no network: stub targets through to an exit code.

These are the acceptance tests. They prove the gate blocks a merge for a known-bad
app, that a quota wall does not, and that per-item values survive a round trip
through SQLite and act as the next run's baseline.
"""
import pytest
from conftest import REFUSAL_POLICY, StubJudge, StubScorer, item

from llm_eval_pipeline.cli import EXIT_GATE_FAILED, EXIT_INCOMPLETE, EXIT_OK, exit_code_for
from llm_eval_pipeline.fixtures.stub_targets import build_stub
from llm_eval_pipeline.paired import PairedConfig
from llm_eval_pipeline.report import markdown_report
from llm_eval_pipeline.runner import Baseline, run_evaluation
from llm_eval_pipeline.aggregate import values_by_item
from llm_eval_pipeline.store import METRIC_COLUMNS, Store
from llm_eval_pipeline.targets.base import TargetUnavailable

# Floors from the spec: deliberately low safety nets.
THRESHOLDS = {
    "faithfulness": {"floor": 0.75, "type": "continuous", "direction": "higher_is_better"},
    "answer_relevance": {"floor": 0.75, "type": "continuous", "direction": "higher_is_better"},
    "abstention_rate": {"floor": 0.70, "type": "binary", "direction": "higher_is_better"},
    "false_refusal_rate": {"floor": 0.20, "type": "binary", "direction": "lower_is_better"},
    "retrieval_hit_rate@k": {"floor": 0.70, "type": "binary", "direction": "higher_is_better"},
    "latency_p95": 10.0,
}

ITEMS = [
    item("a1", gold=[("doc-a", 1)]),
    item("a2", gold=[("doc-a", 1)]),
    item("a3", gold=[("doc-b", 2)]),
    item("u1", kind="unanswerable"),
    item("u2", kind="unanswerable"),
]


def run(target, scorer=None, thresholds=None, baseline=None, items=None, paired_cfg=None, **kwargs):
    return run_evaluation(
        refusal=REFUSAL_POLICY,
        items=items or ITEMS,
        target=target,
        judge=StubJudge(),
        scorer=scorer or StubScorer(),
        thresholds=thresholds or THRESHOLDS,
        retrieval_k=5,
        # Fewer resamples than production: the tests assert verdicts, not precision.
        paired_cfg=paired_cfg or PairedConfig(bootstrap_resamples=500),
        baseline=baseline,
        **kwargs,
    )


class TestRefuseEverything:
    """The required acceptance test from spec 2.3."""

    def test_scores_perfect_abstention_and_total_false_refusal(self):
        outcome = run(build_stub("refuse_everything"))
        assert outcome.summary["abstention_rate"] == 1.0
        assert outcome.summary["false_refusal_rate"] == 1.0

    def test_fails_the_gate_with_exit_code_1(self):
        outcome = run(build_stub("refuse_everything"))
        assert outcome.gate.passed is False
        assert exit_code_for(outcome.gate) == EXIT_GATE_FAILED

    def test_it_is_false_refusal_that_catches_it(self):
        """Abstention alone would have called this app perfect."""
        outcome = run(build_stub("refuse_everything"))
        failed = {c.metric for c in outcome.gate.failures}
        assert "false_refusal_rate" in failed
        abstention = next(c for c in outcome.gate.checks if c.metric == "abstention_rate")
        assert abstention.passed is True


class TestNoRetrieval:
    """The original regression: fluent answers, nothing underneath."""

    def test_unmeasurable_metrics_fail_rather_than_pass(self):
        scorer = StubScorer(faithfulness=None, relevancy=0.988)
        outcome = run(build_stub("no_retrieval"), scorer=scorer)
        assert outcome.gate.passed is False
        failed = {c.metric for c in outcome.gate.failures}
        assert "faithfulness" in failed
        assert "retrieval_hit_rate@5" in failed
        # And the metric that looked fine really did look fine.
        relevance = next(c for c in outcome.gate.checks if c.metric == "answer_relevance")
        assert relevance.passed is True
        assert relevance.actual == pytest.approx(0.988)

    def test_reason_is_recorded_as_not_measured(self):
        outcome = run(build_stub("no_retrieval"), scorer=StubScorer(faithfulness=None))
        check = next(c for c in outcome.gate.checks if c.metric == "retrieval_hit_rate@5")
        assert check.failure_reason == "not measured"


class TestHealthyRun:
    def test_answer_everything_passes_the_floors(self):
        """Never refusing is fine by these floors -- only abstention catches it."""
        outcome = run(
            build_stub("answer_everything"),
            thresholds={k: v for k, v in THRESHOLDS.items() if k != "abstention_rate"},
        )
        assert outcome.gate.passed is True
        assert exit_code_for(outcome.gate) == EXIT_OK

    def test_abstention_floor_catches_answer_everything(self):
        outcome = run(build_stub("answer_everything"))
        assert outcome.gate.passed is False
        assert {c.metric for c in outcome.gate.failures} == {"abstention_rate"}


class TestHardFailureIsNotAQualityVerdict:
    def test_quota_wall_returns_3_not_1(self):
        class OutOfQuota:
            def ask(self, question):
                raise TargetUnavailable("target out of quota: 429")

            def describe(self):
                return {"kind": "stub", "supports_retrieval": True}

        outcome = run(OutOfQuota())
        assert outcome.gate.hard_failure is not None
        assert exit_code_for(outcome.gate) == EXIT_INCOMPLETE

    def test_stops_immediately_rather_than_burning_the_rest(self):
        class FailsOnThird:
            def __init__(self):
                self.asked = 0

            def ask(self, question):
                self.asked += 1
                if self.asked >= 3:
                    raise TargetUnavailable("out of quota")
                return build_stub("answer_everything").ask(question)

            def describe(self):
                return {"kind": "stub", "supports_retrieval": True}

        target = FailsOnThird()
        outcome = run(target)
        assert target.asked == 3  # not 5
        assert len(outcome.results) == 3

    def test_paired_test_is_skipped_on_an_incomplete_run(self):
        """Half a dataset is not comparable to a full baseline."""
        class OutOfQuota:
            def ask(self, question):
                raise TargetUnavailable("429")

            def describe(self):
                return {"kind": "stub"}

        baseline = Baseline(run_id=1, label="base", values={"faithfulness": {"a1": 1.0}})
        outcome = run(OutOfQuota(), baseline=baseline)
        assert outcome.paired.skipped_reason is not None
        assert "did not complete" in outcome.paired.skipped_reason


class TestBaseline:
    def test_no_baseline_skips_the_paired_test_without_failing(self):
        outcome = run(build_stub("answer_everything"), baseline=None)
        assert outcome.paired.skipped_reason is not None
        assert "no comparable baseline" in outcome.paired.skipped_reason
        # Floors alone still decided the run.
        assert outcome.gate.checks

    def test_report_says_the_paired_test_was_skipped(self):
        outcome = run(build_stub("answer_everything"), baseline=None)
        md = markdown_report(
            outcome.summary,
            outcome.gate,
            outcome.results,
            outcome.target_info,
            outcome.judge_model,
            paired=outcome.paired,
        )
        assert "No comparable baseline" in md
        assert "paired test skipped" in md

    def test_disabled_in_config(self):
        outcome = run(
            build_stub("answer_everything"),
            baseline=Baseline(run_id=1, label=None, values={}),
            paired_cfg=PairedConfig(enabled=False),
        )
        assert "disabled" in outcome.paired.skipped_reason

    def test_regression_against_a_stored_baseline_blocks(self, tmp_path):
        """The full loop: run, persist, degrade, compare, block."""
        store = Store(tmp_path / "runs.db")

        good = run(build_stub("answer_everything"), scorer=StubScorer(faithfulness=1.0))
        run_id = store.start_run(
            started_at="2026-01-01T00:00:00+00:00",
            label="baseline",
            dataset="x.json",
            dataset_size=len(ITEMS),
            dataset_hash="hash1",
            judge_config_hash="judge1",
            target={},
            judge_model="stub:judge",
            git_branch="main",
        )
        for record in good.results:
            store.add_result(
                run_id, record, values_by_item(good.item_values).get(record["id"])
            )
        store.finish_run(
            run_id,
            finished_at="2026-01-01T00:01:00+00:00",
            summary=good.summary,
            gate_passed=True,
            gate_detail=[],
        )

        found = store.find_baseline("main", "hash1", "judge1")
        assert found is not None and found["id"] == run_id

        degraded = run(
            build_stub("answer_everything"),
            scorer=StubScorer(faithfulness=0.80),
            baseline=Baseline(run_id=run_id, label="baseline", values=store.item_values(run_id)),
        )
        verdict = degraded.paired.verdicts["faithfulness"]
        assert verdict.regression is True
        # 0.80 clears the 0.75 floor, so only the paired test can catch this.
        assert degraded.summary["faithfulness"] == pytest.approx(0.80)
        assert degraded.gate.passed is False
        assert exit_code_for(degraded.gate) == EXIT_GATE_FAILED

    def test_a_failing_run_is_never_chosen_as_a_baseline(self, tmp_path):
        store = Store(tmp_path / "runs.db")
        run_id = store.start_run(
            started_at="2026-01-01T00:00:00+00:00",
            dataset_hash="h",
            judge_config_hash="j",
            target={},
            git_branch="main",
        )
        store.finish_run(
            run_id,
            finished_at="2026-01-01T00:01:00+00:00",
            summary={},
            gate_passed=False,
            gate_detail=[],
        )
        assert store.find_baseline("main", "h", "j") is None

    def test_an_incomplete_run_is_never_chosen_as_a_baseline(self, tmp_path):
        store = Store(tmp_path / "runs.db")
        run_id = store.start_run(
            started_at="2026-01-01T00:00:00+00:00",
            dataset_hash="h",
            judge_config_hash="j",
            target={},
            git_branch="main",
        )
        store.finish_run(
            run_id,
            finished_at="2026-01-01T00:01:00+00:00",
            summary={},
            gate_passed=False,
            gate_detail=[],
            hard_failure="out of quota",
        )
        row = store.history(1)[0]
        assert row["gate_status"] == "incomplete"
        assert store.find_baseline("main", "h", "j") is None

    @pytest.mark.parametrize(
        "ds_hash,judge_hash",
        [("other", "judge1"), ("hash1", "other"), ("other", "other")],
    )
    def test_a_different_dataset_or_judge_is_not_comparable(self, tmp_path, ds_hash, judge_hash):
        store = Store(tmp_path / "runs.db")
        run_id = store.start_run(
            started_at="2026-01-01T00:00:00+00:00",
            dataset_hash="hash1",
            judge_config_hash="judge1",
            target={},
            git_branch="main",
        )
        store.finish_run(
            run_id,
            finished_at="2026-01-01T00:01:00+00:00",
            summary={},
            gate_passed=True,
            gate_detail=[],
        )
        assert store.find_baseline("main", ds_hash, judge_hash) is None

    def test_a_different_branch_is_not_a_baseline(self, tmp_path):
        store = Store(tmp_path / "runs.db")
        run_id = store.start_run(
            started_at="2026-01-01T00:00:00+00:00",
            dataset_hash="h",
            judge_config_hash="j",
            target={},
            git_branch="feature/x",
        )
        store.finish_run(
            run_id,
            finished_at="2026-01-01T00:01:00+00:00",
            summary={},
            gate_passed=True,
            gate_detail=[],
        )
        assert store.find_baseline("main", "h", "j") is None


class TestStoreRoundTrip:
    def test_item_values_survive_with_nones_intact(self, tmp_path):
        """A None must come back as None. `require_all_metrics` fails the gate on
        an unmeasured metric, and the paired test skips the pair -- both of which
        depend on telling "not measured" apart from "scored zero"."""
        store = Store(tmp_path / "runs.db")
        run_id = store.start_run(started_at="t", target={})
        store.add_result(
            run_id,
            {"id": "a1", "kind": "answerable"},
            {"faithfulness": 0.5, "abstention_rate": None},
        )
        store.finish_run(
            run_id, finished_at="t2", summary={}, gate_passed=True, gate_detail=[]
        )
        values = store.item_values(run_id)
        assert values["faithfulness"] == {"a1": 0.5}
        # A metric with no measurement anywhere is absent, not present-and-None:
        # an all-None row renders in the report as "no pairs", which reads as a
        # fault rather than as "this run did not measure that".
        assert "abstention_rate" not in values

    def test_results_record_retrieval(self, tmp_path):
        store = Store(tmp_path / "runs.db")
        outcome = run(build_stub("answer_everything"))
        run_id = store.start_run(started_at="t", target={})
        by_item = values_by_item(outcome.item_values)
        for record in outcome.results:
            store.add_result(run_id, record, by_item.get(record["id"]))
        rows = store.results_for(run_id)
        assert len(rows) == len(ITEMS)
        assert '"rank": 1' in rows[0]["retrieved"]

    def test_one_row_per_question_in_each_table(self, tmp_path):
        """The shape itself: scores in `results`, evidence in `eval_data`, one
        row per question in each, and no score stored twice."""
        store = Store(tmp_path / "runs.db")
        outcome = run(build_stub("answer_everything"))
        run_id = store.start_run(started_at="t", target={})
        by_item = values_by_item(outcome.item_values)
        for record in outcome.results:
            store.add_result(run_id, record, by_item.get(record["id"]))

        assert len(store.metrics_for(run_id)) == len(ITEMS)
        assert len(store.eval_data_for(run_id)) == len(ITEMS)

        scores = next(r for r in store.metrics_for(run_id) if r["item_id"] == "a1")
        assert scores["faithfulness"] is not None
        assert scores["answer_relevance"] is not None
        # a1 is answerable, so abstention does not apply to it -- NULL, not 0.
        assert scores["abstention_rate"] is None
        assert scores["false_refusal_rate"] is not None

    def test_the_metrics_table_holds_nothing_but_scores(self, tmp_path):
        """`answer` and `contexts` run to kilobytes. Keeping them out is the
        reason the two tables exist."""
        store = Store(tmp_path / "runs.db")
        outcome = run(build_stub("answer_everything"))
        run_id = store.start_run(started_at="t", target={})
        store.add_result(run_id, outcome.results[0], {"faithfulness": 0.9})
        row = store.metrics_for(run_id)[0]
        assert set(row) == {"id", "run_id", "item_id", *METRIC_COLUMNS}

    def test_the_evidence_table_holds_no_scores(self, tmp_path):
        """The other half: a score has exactly one home."""
        store = Store(tmp_path / "runs.db")
        outcome = run(build_stub("answer_everything"))
        run_id = store.start_run(started_at="t", target={})
        store.add_result(run_id, outcome.results[0], {"faithfulness": 0.9})
        row = store.eval_data_for(run_id)[0]
        for column in METRIC_COLUMNS:
            assert column not in row, f"{column} belongs in `results`"
        assert row["question"] and row["answer"]

    def test_results_for_joins_both_tables(self, tmp_path):
        """A human debugging a bad score wants the answer next to it."""
        store = Store(tmp_path / "runs.db")
        outcome = run(build_stub("answer_everything"))
        run_id = store.start_run(started_at="t", target={})
        store.add_result(run_id, outcome.results[0], {"faithfulness": 0.9})
        row = store.results_for(run_id)[0]
        assert row["faithfulness"] == 0.9
        assert row["question"] and row["answer"]

    def test_the_retrieval_column_carries_no_k(self, tmp_path):
        """`retrieval_hit_rate@5` is the metric NAME; the column is
        `retrieval_hit_rate`. k belongs to the run, so changing it must not
        require an ALTER TABLE."""
        store = Store(tmp_path / "runs.db")
        outcome = run(build_stub("answer_everything"))
        run_id = store.start_run(started_at="t", target={})
        by_item = values_by_item(outcome.item_values)
        for record in outcome.results:
            store.add_result(run_id, record, by_item.get(record["id"]))
        row = store.metrics_for(run_id)[0]
        assert "retrieval_hit_rate" in row
        assert "retrieval_hit_rate@5" not in row

    def test_k_is_restored_onto_the_metric_name_when_read_back(self, tmp_path):
        """The gate looks metrics up by name, so the round trip has to put `@k`
        back -- a hit rate at 5 is not the same measurement as one at 20."""
        store = Store(tmp_path / "runs.db")
        run_id = store.start_run(started_at="t", target={})
        store.add_result(
            run_id, {"id": "a1", "kind": "answerable"}, {"retrieval_hit_rate@5": 1.0}
        )
        store.finish_run(
            run_id,
            finished_at="t2",
            summary={"retrieval_k": 5},
            gate_passed=True,
            gate_detail=[],
        )
        assert store.item_values(run_id) == {"retrieval_hit_rate@5": {"a1": 1.0}}

    def test_a_baseline_round_trips_through_the_wide_table(self, tmp_path):
        """The real requirement: what the paired test reads back must match what
        the run computed, or a regression would be measured against a different
        quantity than the one that was stored."""
        store = Store(tmp_path / "runs.db")
        outcome = run(build_stub("answer_everything"))
        run_id = store.start_run(started_at="t", target={})
        by_item = values_by_item(outcome.item_values)
        for record in outcome.results:
            store.add_result(run_id, record, by_item.get(record["id"]))
        store.finish_run(
            run_id,
            finished_at="t2",
            summary=outcome.summary,
            gate_passed=True,
            gate_detail=[],
        )

        stored = store.item_values(run_id)
        for metric, per_item in outcome.item_values.items():
            if all(v is None for v in per_item.values()):
                continue  # dropped on purpose; see the test above
            assert stored[metric] == per_item, metric

    def test_an_old_database_is_folded_in_without_losing_scores(self, tmp_path):
        """Run history cannot be regenerated -- the paired gate's baseline IS
        these rows -- so dropping the old long-format table must happen only
        after its numbers have been copied across."""
        import sqlite3

        path = tmp_path / "legacy.db"
        with sqlite3.connect(path) as conn:
            conn.executescript(
                """
                CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                   started_at TEXT NOT NULL, summary TEXT);
                CREATE TABLE results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL, item_id TEXT NOT NULL,
                    question TEXT NOT NULL, answer TEXT,
                    faithfulness REAL, relevancy REAL, hallucinated INTEGER,
                    detail TEXT);
                CREATE TABLE item_results (
                    run_id INTEGER NOT NULL, item_id TEXT NOT NULL,
                    metric TEXT NOT NULL, value REAL,
                    PRIMARY KEY (run_id, metric, item_id));
                INSERT INTO runs (started_at, summary)
                    VALUES ('2026-01-01T00:00:00+00:00', '{"retrieval_k": 5}');
                INSERT INTO results
                    (run_id, item_id, question, answer, faithfulness, relevancy,
                     hallucinated, detail)
                    VALUES (1, 'a1', 'q?', 'an answer', 0.91, 0.88, 1,
                            '{"reasons": {"faithfulness": "claim 2 unsupported"}}');
                INSERT INTO item_results VALUES (1, 'a1', 'abstention_rate', 1.0);
                INSERT INTO item_results VALUES (1, 'a1', 'retrieval_hit_rate@5', 1.0);
                """
            )

        store = Store(path)  # opening it migrates

        # The old per-question table became `eval_data`, rows intact.
        data = store.eval_data_for(1)[0]
        assert data["question"] == "q?" and data["answer"] == "an answer"
        assert "claim 2 unsupported" in (data["reasons"] or "")
        # Every score column was moved out of it.
        for column in ("faithfulness", "relevancy", "hallucinated", "detail"):
            assert column not in data, f"{column} should have moved to `results`"

        # And every score landed in `results`.
        scores = store.metrics_for(1)[0]
        assert scores["faithfulness"] == 0.91
        # `relevancy` was renamed, not dropped...
        assert scores["answer_relevance"] == 0.88
        # ...the 0/1 flag became the rate the gate actually reads...
        assert scores["hallucination_rate"] == 1.0
        # ...and the long-format rows were folded in too.
        assert scores["abstention_rate"] == 1.0
        assert scores["retrieval_hit_rate"] == 1.0

    def test_folding_in_is_idempotent(self, tmp_path):
        """Every open re-runs the migration, so a second one must be a no-op."""
        store = Store(tmp_path / "x.db")
        run_id = store.start_run(started_at="t", target={})
        store.add_result(run_id, {"id": "a1"}, {"faithfulness": 0.5})
        store.finish_run(
            run_id, finished_at="t2", summary={}, gate_passed=True, gate_detail=[]
        )
        again = Store(tmp_path / "x.db")
        assert again.item_values(run_id) == {"faithfulness": {"a1": 0.5}}

    def test_migration_adds_columns_to_an_old_database(self, tmp_path):
        """Run history cannot be regenerated, so an upgrade must not require a wipe."""
        import sqlite3

        path = tmp_path / "old.db"
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE runs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "started_at TEXT NOT NULL, label TEXT)"
            )
            conn.execute("INSERT INTO runs (started_at, label) VALUES ('t', 'legacy')")
        store = Store(path)
        rows = store.history()
        assert rows[0]["label"] == "legacy"  # old row preserved
        assert "dataset_hash" in rows[0]  # new column present
