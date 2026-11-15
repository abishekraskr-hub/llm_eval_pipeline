"""Backend selection, branch-name -> identifier mapping, and a Postgres round trip.

Everything except the final class runs offline with no driver installed. The
Postgres class skips itself when no server answers, so the suite stays runnable
on a machine that has never had Postgres on it -- which is the whole reason CI
can still gate on these tests.
"""
import os
from pathlib import Path

import pytest

# The library ships no config of its own: configs belong to apps. The bundled
# example doubles as the fixture, which also keeps the quickstart honest --
# if it drifts, these tests fail.
EXAMPLE_CONFIG = (
    Path(__file__).resolve().parent.parent / "examples" / "echo_target" / "llmeval.yaml"
)

from llm_eval_pipeline.config import DEFAULTS, load_config
from llm_eval_pipeline.paired import PairedConfig
from llm_eval_pipeline.storage import (
    BACKENDS,
    StoreUnavailable,
    describe_params,
    open_store,
    postgres_params,
    resolve_backend,
)
from llm_eval_pipeline.store import Store
from llm_eval_pipeline.store_pg import (
    MAX_SLUG,
    branch_slugs,
    driver_available,
    migrate_sql,
    slugify_branch,
)

# `importorskip('psycopg')` is NOT enough: psycopg is routinely installed and
# still unimportable because no libpq can be loaded. `driver_available()` goes
# through the platform's own loader, which finds an installed PostgreSQL's
# libpq first -- so these tests run wherever a real run would.
needs_driver = pytest.mark.skipif(
    not driver_available(), reason="psycopg/libpq not loadable here"
)


class TestBranchSlug:
    @pytest.mark.parametrize(
        "branch,expected",
        [
            ("develop", "develop"),
            ("main", "main"),
            ("feature/retrieval-k", "feature_retrieval_k"),
            ("fix/judge--timeout", "fix_judge_timeout"),
            ("release/v1.2.0", "release_v1_2_0"),
            ("UPPER/Case", "upper_case"),
            ("trailing/", "trailing"),
            ("/leading", "leading"),
        ],
    )
    def test_readable_names_survive(self, branch, expected):
        assert slugify_branch(branch) == expected

    def test_a_pr_ref_does_not_start_with_a_digit(self):
        """`42/merge` is what GITHUB_REF_NAME gives on a pull_request event.

        An identifier beginning with a digit is legal only while quoted, and
        sooner or later someone types it unquoted into psql.
        """
        assert slugify_branch("42/merge") == "b_42_merge"

    @pytest.mark.parametrize("branch", ["", "///", "---", None])
    def test_a_nameless_branch_still_gets_an_identifier(self, branch):
        assert slugify_branch(branch) == "unknown"

    def test_a_long_branch_name_leaves_room_for_the_items_suffix(self):
        """Postgres truncates at 63 bytes, and the item view appends `__items`.

        Without the reserved room, two different long branch names would be
        truncated to the SAME view name and silently share one window onto the
        data -- the worst possible failure for a per-branch report.
        """
        slug = slugify_branch("feature/" + "x" * 200)
        assert len(slug) <= MAX_SLUG
        assert len(slug + "__items") <= 63


class TestBranchSlugCollisions:
    def test_names_that_slugify_the_same_get_distinct_identifiers(self):
        mapping = branch_slugs(["feature/x", "feature-x", "Feature/X"])
        assert len(set(mapping.values())) == 3

    def test_the_mapping_does_not_depend_on_input_order(self):
        branches = ["feature/x", "feature-x", "develop"]
        assert branch_slugs(branches) == branch_slugs(list(reversed(branches)))

    def test_one_of_the_colliding_names_keeps_the_readable_form(self):
        """Disambiguation is only applied to the later name, so the common case
        stays legible instead of every view carrying a hash."""
        mapping = branch_slugs(["feature/x", "feature-x"])
        assert "feature_x" in mapping.values()

    def test_adding_an_unrelated_branch_does_not_rename_existing_views(self):
        before = branch_slugs(["develop", "feature/x"])
        after = branch_slugs(["develop", "feature/x", "hotfix/y"])
        assert all(after[b] == slug for b, slug in before.items())

    def test_blank_branches_are_dropped_rather_than_named_unknown(self):
        assert branch_slugs(["develop", "", None]) == {"develop": "develop"}

    def test_disambiguated_names_also_fit(self):
        long_a, long_b = "feature/" + "x" * 200, "feature-" + "x" * 200
        for slug in branch_slugs([long_a, long_b]).values():
            assert len(slug + "__items") <= 63


class TestBackendSelection:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """EVAL_STORAGE_BACKEND in the ambient environment -- which CI sets --
        must not decide whether these pass. The tests that care set it."""
        monkeypatch.delenv("EVAL_STORAGE_BACKEND", raising=False)

    def test_the_config_default_is_postgres(self):
        """The project's run history lives in Postgres; sqlite is the fallback."""
        assert DEFAULTS["storage"]["backend"] == "postgres"
        assert DEFAULTS["storage"]["postgres"]["dbname"] == "llm_eval_pipeline"

    def test_the_password_is_never_in_the_config(self):
        pg = DEFAULTS["storage"]["postgres"]
        assert "password" not in pg
        assert pg["password_env"] == "EVAL_PG_PASSWORD"

    def test_an_explicit_flag_beats_the_config(self):
        cfg = {"storage": {"backend": "postgres"}}
        assert resolve_backend(cfg, "sqlite") == "sqlite"

    def test_the_environment_beats_the_config(self, monkeypatch):
        monkeypatch.setenv("EVAL_STORAGE_BACKEND", "sqlite")
        assert resolve_backend({"storage": {"backend": "postgres"}}) == "sqlite"

    def test_the_flag_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("EVAL_STORAGE_BACKEND", "sqlite")
        assert resolve_backend({}, "postgres") == "postgres"

    def test_an_unknown_backend_is_rejected_by_name(self):
        with pytest.raises(StoreUnavailable) as exc:
            resolve_backend({"storage": {"backend": "mysql"}})
        assert "mysql" in str(exc.value)
        for backend in BACKENDS:
            assert backend in str(exc.value)

    def test_sqlite_opens_a_real_store(self, tmp_path):
        cfg = {"storage": {"backend": "sqlite", "sqlite_path": str(tmp_path / "x.db")}}
        store = open_store(cfg)
        assert isinstance(store, Store)
        assert store.backend == "sqlite"
        assert store.history() == []

    def test_sqlite_path_is_resolved_from_the_config_folder(self, tmp_path):
        """Not from the cwd. A config that worked locally and broke in CI was the
        failure this rule exists to prevent."""
        cfg = {
            "storage": {"backend": "sqlite", "sqlite_path": "sub/history.sqlite"},
            "_dir": str(tmp_path),
        }
        assert open_store(cfg).path == tmp_path / "sub" / "history.sqlite"

    def test_sqlite_has_no_branch_views_but_answers_anyway(self, tmp_path):
        """The no-op keeps the caller from having to ask which backend it holds."""
        store = open_store({"storage": {"backend": "sqlite", "sqlite_path": str(tmp_path / "a.db")}})
        assert store.refresh_branch_views() == {}


class TestPostgresParams:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """A developer's own EVAL_PG_* vars must not decide whether this passes.

        These tests are about what the CONFIG says; the override tests below set
        what they need explicitly.
        """
        for var in (
            "EVAL_DATABASE_URL",
            "EVAL_PG_HOST",
            "EVAL_PG_PORT",
            "EVAL_PG_DBNAME",
            "EVAL_PG_USER",
            "EVAL_PG_PASSWORD",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_the_config_supplies_the_connection(self):
        params = postgres_params(DEFAULTS)
        assert params["dbname"] == "llm_eval_pipeline"
        assert params["user"] == "postgres"
        assert params["schema"] == "eval"
        assert params["branch_schema"] == "branches"

    def test_the_password_comes_from_the_named_env_var(self, monkeypatch):
        monkeypatch.setenv("EVAL_PG_PASSWORD", "s3cret")
        assert postgres_params(DEFAULTS)["password"] == "s3cret"

    def test_a_password_with_an_at_sign_is_passed_through_untouched(self, monkeypatch):
        """The trap this whole code path exists to avoid.

        libpq keyword parameters take the password literally, so `abi@123` needs
        no escaping. Inside a URL DSN the same value would have to be written
        `abi%40123` or the `@` would be read as the host separator.
        """
        monkeypatch.setenv("EVAL_PG_PASSWORD", "abi@123")
        assert postgres_params(DEFAULTS)["password"] == "abi@123"

    def test_a_full_dsn_clears_the_individual_settings(self, monkeypatch):
        """Otherwise host/dbname from the config would override parts of the DSN."""
        monkeypatch.setenv("EVAL_DATABASE_URL", "postgresql://u:p@example:5432/db")
        params = postgres_params(DEFAULTS)
        assert params["conninfo"] == "postgresql://u:p@example:5432/db"
        assert all(params[k] is None for k in ("host", "port", "dbname", "user", "password"))

    def test_env_vars_override_the_config(self, monkeypatch):
        monkeypatch.setenv("EVAL_PG_HOST", "db.internal")
        monkeypatch.setenv("EVAL_PG_DBNAME", "other")
        params = postgres_params(DEFAULTS)
        assert (params["host"], params["dbname"]) == ("db.internal", "other")

    def test_the_printable_summary_never_leaks_the_password(self, monkeypatch):
        monkeypatch.setenv("EVAL_PG_PASSWORD", "abi@123")
        assert "abi@123" not in describe_params(postgres_params(DEFAULTS))

    def test_a_dsn_summary_names_the_variable_not_its_value(self, monkeypatch):
        monkeypatch.setenv("EVAL_DATABASE_URL", "postgresql://u:hunter2@h/db")
        text = describe_params(postgres_params(DEFAULTS))
        assert "hunter2" not in text and "EVAL_DATABASE_URL" in text


@needs_driver
class TestPostgresFailsLoudly:
    def test_an_unreachable_server_raises_rather_than_falling_back(self, monkeypatch):
        """No silent fallback to SQLite, by design.

        A fallback would report a green run whose rows are in a file nobody went
        looking in -- and would let the paired gate compare against a baseline
        from a different database.
        """
        monkeypatch.delenv("EVAL_DATABASE_URL", raising=False)
        monkeypatch.setenv("EVAL_PG_HOST", "127.0.0.1")
        monkeypatch.setenv("EVAL_PG_PORT", "1")
        with pytest.raises(StoreUnavailable) as exc:
            open_store(DEFAULTS, backend="postgres")
        message = str(exc.value)
        assert "--storage sqlite" in message, "the message must say how to proceed"
        assert "127.0.0.1" in message, "the message must say what it tried"


class TestMigrationIsGenerated:
    def test_every_optional_column_can_be_added_to_an_old_database(self):
        statements = migrate_sql()
        assert any("gate_status" in s for s in statements)
        assert any("baseline_run_id" in s for s in statements)
        assert all("IF NOT EXISTS" in s for s in statements)

    def test_keys_and_not_null_columns_are_skipped(self):
        """They cannot be bolted onto a populated table, and have always existed."""
        statements = " ".join(migrate_sql())
        assert "ADD COLUMN IF NOT EXISTS id " not in statements
        assert "ADD COLUMN IF NOT EXISTS metric " not in statements


class TestBaselineBranch:
    def test_the_default_integration_branch_is_develop(self):
        assert PairedConfig().base_branch == "develop"
        assert DEFAULTS["paired"]["base_branch"] == "develop"

    def test_the_shipped_config_agrees_with_the_default(self):
        assert load_config(EXAMPLE_CONFIG)["paired"]["base_branch"] == "develop"

    def test_the_env_var_can_override_it(self, monkeypatch):
        monkeypatch.setenv("LLMEVAL_BASE_BRANCH", "release/2026-11")
        assert load_config(EXAMPLE_CONFIG)["paired"]["base_branch"] == "release/2026-11"


# ---------------------------------------------------------------------------
# Live Postgres. Skipped unless a server answers.
# ---------------------------------------------------------------------------

def _admin_conn(cfg: dict):
    """An autocommit connection to the maintenance database, or None."""
    from llm_eval_pipeline.storage import MAINTENANCE_DB
    from llm_eval_pipeline.store_pg import _psycopg

    psycopg, _, dict_row, _j = _psycopg()
    params = postgres_params(cfg)
    admin = {
        k: params[k]
        for k in ("host", "port", "user", "password", "connect_timeout")
        if params.get(k) is not None
    }
    admin["dbname"] = MAINTENANCE_DB
    try:
        return psycopg.connect(autocommit=True, row_factory=dict_row, **admin)
    except psycopg.Error:
        return None


def _server_reachable(cfg: dict) -> bool:
    conn = _admin_conn(cfg)
    if conn is None:
        return False
    conn.close()
    return True


def _create_database_raw(cfg: dict, name: str) -> None:
    from llm_eval_pipeline.store_pg import _psycopg

    _psycopg_mod, sql, _dr, _j = _psycopg()
    conn = _admin_conn(cfg)
    try:
        conn.execute(sql.SQL("CREATE DATABASE {db}").format(db=sql.Identifier(name)))
    except Exception:
        pass
    finally:
        conn.close()


def _drop_database(cfg: dict, name: str) -> None:
    from llm_eval_pipeline.store_pg import _psycopg

    _psycopg_mod, sql, _dr, _j = _psycopg()
    conn = _admin_conn(cfg)
    if conn is None:
        return
    try:
        # Evict our own leftover connections first, or the DROP is refused.
        conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = %s AND pid <> pg_backend_pid()",
            (name,),
        )
        conn.execute(sql.SQL("DROP DATABASE IF EXISTS {db}").format(db=sql.Identifier(name)))
    except Exception:
        pass
    finally:
        conn.close()


def _live_store(schema: str):
    import psycopg

    from llm_eval_pipeline.store_pg import PostgresStore

    params = postgres_params(load_config(EXAMPLE_CONFIG))
    params["schema"] = schema
    params["branch_schema"] = f"{schema}_branches"
    try:
        return PostgresStore(**params)
    except StoreUnavailable as exc:
        pytest.skip(f"no Postgres to test against: {exc}")
    except psycopg.Error as exc:  # pragma: no cover
        pytest.skip(f"no Postgres to test against: {exc}")


@needs_driver
@pytest.mark.skipif(
    os.getenv("EVAL_SKIP_PG_TESTS") == "1", reason="EVAL_SKIP_PG_TESTS=1"
)
class TestLivePostgres:
    """Runs against a throwaway schema, so it cannot touch real run history."""

    @pytest.fixture
    def store(self):
        s = _live_store("eval_test_tmp")
        yield s
        with s._cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS eval_test_tmp CASCADE")
            cur.execute("DROP SCHEMA IF EXISTS eval_test_tmp_branches CASCADE")
        s.close()

    def test_a_run_round_trips(self, store):
        run_id = store.start_run(
            started_at="2026-01-01T00:00:00+00:00",
            label="t",
            dataset_hash="h",
            judge_config_hash="j",
            target={"kind": "stub"},
            git_branch="develop",
        )
        store.add_result(run_id, {"id": "a1", "kind": "answerable"}, {"faithfulness": 0.9})
        store.add_result(run_id, {"id": "u1", "kind": "unanswerable"}, {"faithfulness": None})
        store.finish_run(
            run_id,
            finished_at="2026-01-01T00:01:00+00:00",
            summary={"faithfulness": 0.9},
            gate_passed=True,
            gate_detail=[],
        )
        # None survives the round trip: "not measured" is not "scored zero".
        assert store.item_values(run_id) == {"faithfulness": {"a1": 0.9, "u1": None}}
        row = store.history(1)[0]
        assert row["gate_status"] == "pass"
        assert row["summary"] == {"faithfulness": 0.9}
        assert row["target"] == {"kind": "stub"}
        # Timestamps are flattened to ISO strings, as the SQLite store returns
        # them: the CLI and the reports slice `started_at` as text.
        assert isinstance(row["started_at"], str)

    def test_every_metric_is_a_column_on_one_row(self, store):
        """One row per question, no score stored twice."""
        run_id = store.start_run(started_at="2026-01-01T00:00:00+00:00", target={})
        store.add_result(
            run_id,
            {"id": "a1", "kind": "answerable", "question": "q?", "answer": "a"},
            {"faithfulness": 0.5, "answer_relevance": 0.9, "abstention_rate": None},
        )
        rows = store.metrics_for(run_id)
        assert len(rows) == 1
        row = rows[0]
        assert (row["faithfulness"], row["answer_relevance"]) == (0.5, 0.9)
        assert row["abstention_rate"] is None
        # The evidence is in the other table, keyed the same way.
        assert store.eval_data_for(run_id)[0]["question"] == "q?"

    def test_a_feature_branch_finds_the_develop_baseline(self, store):
        base = store.start_run(
            started_at="2026-01-01T00:00:00+00:00",
            dataset_hash="h",
            judge_config_hash="j",
            target={},
            git_branch="develop",
        )
        store.finish_run(
            base,
            finished_at="2026-01-01T00:01:00+00:00",
            summary={},
            gate_passed=True,
            gate_detail=[],
        )
        current = store.start_run(
            started_at="2026-01-02T00:00:00+00:00",
            dataset_hash="h",
            judge_config_hash="j",
            target={},
            git_branch="feature/retrieval-k",
        )
        found = store.find_baseline("develop", "h", "j", exclude_run_id=current)
        assert found is not None and found["id"] == base

    def test_a_failing_develop_run_is_never_the_baseline(self, store):
        run_id = store.start_run(
            started_at="2026-01-01T00:00:00+00:00",
            dataset_hash="h",
            judge_config_hash="j",
            target={},
            git_branch="develop",
        )
        store.finish_run(
            run_id,
            finished_at="2026-01-01T00:01:00+00:00",
            summary={},
            gate_passed=False,
            gate_detail=[],
        )
        assert store.find_baseline("develop", "h", "j") is None

    def test_a_null_hash_matches_a_null_hash(self, store):
        """`=` never matches NULL in SQL; `IS NOT DISTINCT FROM` does.

        A run recorded before hashing existed has NULL hashes, and must still be
        findable rather than silently invisible.
        """
        run_id = store.start_run(
            started_at="2026-01-01T00:00:00+00:00", target={}, git_branch="develop"
        )
        store.finish_run(
            run_id,
            finished_at="2026-01-01T00:01:00+00:00",
            summary={},
            gate_passed=True,
            gate_detail=[],
        )
        assert store.find_baseline("develop", None, None)["id"] == run_id

    def test_each_branch_gets_a_pair_of_views(self, store):
        for branch in ("develop", "feature/retrieval-k"):
            run_id = store.start_run(
                started_at="2026-01-01T00:00:00+00:00",
                target={},
                git_branch=branch,
                dataset_size=10,
            )
            store.add_result(run_id, {"id": "a1"}, {"faithfulness": 0.9})
            store.finish_run(
                run_id,
                finished_at="2026-01-01T00:01:00+00:00",
                summary={"faithfulness": 0.9, "latency_p95": 2.0},
                gate_passed=True,
                gate_detail=[],
            )
        mapping = store.refresh_branch_views()
        assert mapping == {"develop": "develop", "feature/retrieval-k": "feature_retrieval_k"}

        with store._cursor() as cur:
            cur.execute("SELECT * FROM eval_test_tmp_branches.feature_retrieval_k")
            rows = cur.fetchall()
            assert len(rows) == 1 and rows[0]["faithfulness"] == 0.9
            assert rows[0]["questions"] == 10
            # The view is scoped to its own branch and nothing else.
            cur.execute("SELECT count(*) AS n FROM eval_test_tmp_branches.develop")
            assert cur.fetchone()["n"] == 1
            cur.execute(
                "SELECT item_id, faithfulness "
                "FROM eval_test_tmp_branches.feature_retrieval_k__items"
            )
            assert cur.fetchall() == [{"item_id": "a1", "faithfulness": 0.9}]

    def test_refreshing_twice_is_safe_even_when_the_columns_change(self, store):
        """DROP+CREATE, not CREATE OR REPLACE: replacing a view fails outright if
        its column list moved, which happens whenever a metric is added."""
        run_id = store.start_run(
            started_at="2026-01-01T00:00:00+00:00", target={}, git_branch="develop"
        )
        store.finish_run(
            run_id,
            finished_at="2026-01-01T00:01:00+00:00",
            summary={},
            gate_passed=True,
            gate_detail=[],
        )
        assert store.refresh_branch_views() == store.refresh_branch_views()

    def test_views_for_branches_with_no_runs_are_dropped(self, store):
        run_id = store.start_run(
            started_at="2026-01-01T00:00:00+00:00", target={}, git_branch="gone"
        )
        store.refresh_branch_views()
        with store._cursor() as cur:
            cur.execute("DELETE FROM eval_test_tmp.runs WHERE id = %s", (run_id,))
        store.refresh_branch_views()
        with store._cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.views WHERE table_schema = %s",
                ("eval_test_tmp_branches",),
            )
            assert [r["table_name"] for r in cur.fetchall()] == []

    def test_opening_an_existing_database_twice_does_not_fail(self, store):
        """The migration has to be idempotent: every run re-applies it."""
        second = _live_store("eval_test_tmp")
        second.close()


class TestDryRunCannotPoisonTheBaseline:
    """`--dry-run` makes the pipeline free to exercise. The price of that is that
    its numbers must never be able to masquerade as a measurement."""

    def test_no_contexts_means_unmeasured_not_zero(self):
        from llm_eval_pipeline.fixtures.dry_run import DryRunScorer

        scores = DryRunScorer().score("q", "an answer", [])
        assert scores.faithfulness is None, "0.0 would read as a measured failure"
        assert scores.hallucinated is None
        assert "faithfulness" in scores.errors

    def test_scores_are_produced_when_there_are_contexts(self):
        from llm_eval_pipeline.fixtures.dry_run import DryRunScorer

        scores = DryRunScorer().score("q", "an answer", ["some context"])
        assert 0.0 < scores.faithfulness <= 1.0
        assert "dry run" in scores.reasons["faithfulness"]

    def test_the_judge_makes_no_calls(self):
        from llm_eval_pipeline.fixtures.dry_run import DryRunJudge, DryRunScorer

        judge = DryRunJudge()
        scorer = DryRunScorer(judge)
        for _ in range(5):
            scorer.score("q", "a", ["c"])
        assert judge.calls == 0
        assert "dryrun" in judge.get_model_name()

    def test_a_tagged_judge_hash_can_never_match_a_real_one(self, tmp_path):
        """This single fact is what keeps a dry run out of the baseline, in both
        directions, with no special case anywhere in the gate: baseline selection
        requires an exactly matching judge_config_hash."""
        store = Store(tmp_path / "x.db")
        real = store.start_run(
            started_at="2026-01-01T00:00:00+00:00",
            dataset_hash="h",
            judge_config_hash="abc123",
            target={},
            git_branch="develop",
        )
        store.finish_run(
            real,
            finished_at="2026-01-01T00:01:00+00:00",
            summary={},
            gate_passed=True,
            gate_detail=[],
        )
        # A dry run looking for a baseline finds nothing...
        assert store.find_baseline("develop", "h", "dryrun-abc123") is None
        # ...and a real run would not accept a dry run's rows either.
        assert store.find_baseline("develop", "h", "abc123")["id"] == real


class TestMissingDatabaseDetection:
    def test_libpq_s_missing_database_message_is_recognised(self):
        from llm_eval_pipeline.storage import _is_missing_database

        assert _is_missing_database(
            Exception('connection failed: FATAL:  database "x" does not exist')
        )

    def test_other_failures_are_not_mistaken_for_it(self):
        from llm_eval_pipeline.storage import _is_missing_database

        for message in (
            "connection refused",
            "password authentication failed for user",
            "timeout expired",
        ):
            assert not _is_missing_database(Exception(message))


@needs_driver
class TestCreateDatabasePolicy:
    """What happens when the database is absent, without needing it to be."""

    @pytest.fixture(autouse=True)
    def _absent(self, monkeypatch):
        # A reachable server is required, not merely an installed driver: with no
        # credentials the connect fails on AUTH, so the "does not exist" branch
        # these tests assert on is never reached.
        if not _server_reachable(load_config(EXAMPLE_CONFIG)):
            pytest.skip("no reachable Postgres")
        monkeypatch.delenv("EVAL_DATABASE_URL", raising=False)
        monkeypatch.setenv("EVAL_PG_DBNAME", "eval_absent_db_for_tests")

    def _cfg(self, **pg):
        cfg = {k: v for k, v in DEFAULTS.items()}
        cfg["storage"] = {
            "backend": "postgres",
            "postgres": {**DEFAULTS["storage"]["postgres"], **pg},
        }
        return cfg

    def test_no_create_database_fails_with_the_sql_to_run(self):
        """The whole point of the opt-out: a typo must not become a database."""
        with pytest.raises(StoreUnavailable) as exc:
            open_store(self._cfg(), backend="postgres", create_db=False)
        message = str(exc.value)
        assert "CREATE DATABASE" in message
        assert "eval_absent_db_for_tests" in message
        assert "--create-database" in message, "must say how to proceed"

    def test_a_dsn_is_never_auto_created(self, monkeypatch):
        """The name lives inside the DSN, and a hosted Postgres provisions it."""
        monkeypatch.setenv(
            "EVAL_DATABASE_URL",
            "postgresql://postgres@127.0.0.1:5432/eval_absent_db_for_tests",
        )
        with pytest.raises(StoreUnavailable) as exc:
            open_store(self._cfg(), backend="postgres", create_db=True)
        assert "EVAL_DATABASE_URL" in str(exc.value)

    def test_the_flag_overrides_a_config_that_says_no(self, monkeypatch):
        """create_db=None defers to the config; True/False override it."""
        cfg = self._cfg(create_database=False)
        with pytest.raises(StoreUnavailable) as exc:
            open_store(cfg, backend="postgres", create_db=None)
        assert "--create-database" in str(exc.value)


@needs_driver
@pytest.mark.skipif(os.getenv("EVAL_SKIP_PG_TESTS") == "1", reason="EVAL_SKIP_PG_TESTS=1")
class TestCreateDatabaseForReal:
    DB = "eval_created_by_tests"

    @pytest.fixture
    def cfg(self, monkeypatch):
        monkeypatch.delenv("EVAL_DATABASE_URL", raising=False)
        monkeypatch.setenv("EVAL_PG_DBNAME", self.DB)
        base = load_config(EXAMPLE_CONFIG)
        if not _server_reachable(base):
            pytest.skip("no Postgres to test against")
        yield base
        _drop_database(base, self.DB)
        _drop_database(base, " " + self.DB)

    def test_a_missing_database_is_created_and_usable(self, cfg):
        notices: list[str] = []
        store = open_store(cfg, backend="postgres", create_db=True, notify=notices.append)
        try:
            assert any("created database" in n for n in notices)
            # And it is immediately usable: the schema went in on the same open.
            assert store.history() == []
            run_id = store.start_run(started_at="2026-01-01T00:00:00+00:00", target={})
            assert run_id == 1
        finally:
            store.close()

    def test_opening_it_again_creates_nothing_and_says_nothing(self, cfg):
        open_store(cfg, backend="postgres", create_db=True).close()
        notices: list[str] = []
        open_store(cfg, backend="postgres", create_db=True, notify=notices.append).close()
        assert notices == [], "a second open must be silent"

    def test_a_near_miss_name_is_warned_about_loudly(self, cfg):
        """The pgAdmin leading-space slip, reproduced.

        Creating the correctly-spelled database is still the right move -- but
        doing it silently would leave two databases and no clue which holds the
        rows, which is exactly what happened on this project once already.
        """
        _create_database_raw(cfg, " " + self.DB)
        notices: list[str] = []
        open_store(cfg, backend="postgres", create_db=True, notify=notices.append).close()
        text = "\n".join(notices)
        assert "WARNING" in text
        assert f"' {self.DB}'" in text, "the warning must name the near miss"
        assert "dbname" in text, "and point at the setting to check"
