"""Run history in PostgreSQL.

Same public API as `store.Store`, so the runner and the CLI never learn which
backend they are talking to. SQLite stays the offline/CI fallback; Postgres is
for a real database you can open in pgAdmin, point a dashboard at, or query by
hand.

Layout inside the database
--------------------------

    eval.runs              one row per evaluation
    eval.eval_data         one row per QUESTION: what was asked, what came back,
                           what it cost, and the judge's reasoning
    eval.results           one row per QUESTION: nothing but the metric scores,
                           one column each
    eval.branch_overview   one row per git branch: counts and last passing run

`eval_data` and `results` are keyed the same way, `(run_id, item_id)`.

Each score is written in exactly ONE place: `results`. Nothing else holds a
score, so no two copies of a number can disagree -- which is what happened when
a second long-format table held them all again.

The split keeps `results` narrow and numeric. `answer` and `contexts` run to
kilobytes per row, so a `SELECT *` over the scores would otherwise drag the
whole corpus along with it.

Note on the retrieval columns: the metric is NAMED `retrieval_hit_rate@5`, but
the COLUMN is `retrieval_hit_rate`. `k` belongs to the run, recorded once in
`runs.summary`, not to each row -- baking it into a column name would mean an
ALTER TABLE every time `retrieval.k` changed.

    branches."<branch>"           that branch's runs, key metrics as columns
    branches."<branch>__items"    that branch's per-question metric values

Why `branches` holds VIEWS and not one real table per branch
------------------------------------------------------------

A view reads exactly like a table in pgAdmin, which is the whole point of asking
for one table per branch -- but it cannot drift from the data, and it keeps four
properties that real per-branch tables would break:

1. The paired gate joins a run against its baseline on `item_id`. A feature
   branch and its `develop` baseline would then live in DIFFERENT tables, so the
   core query of the gate would have to be assembled from a branch name at
   runtime. Every comparison would mean dynamic SQL built from a git ref.
2. Writing a result on a new branch would need `CREATE TABLE` during the eval
   run. DDL in the hot path needs elevated rights and turns a typo in a branch
   name into a permanent schema object.
3. `runs.baseline_run_id` points at a run by id, so ids have to be unique across
   all branches. Per-branch tables would need a shared sequence anyway -- which
   is the single `runs` table, wearing a disguise.
4. "How did this metric move across branches?" is one `GROUP BY` here and a
   UNION over N unknown tables there.

So the data has one home and each branch gets its own window onto it.
"""
import glob
import hashlib
import os
import re
import sys
from contextlib import contextmanager

from .store import _LEGACY_SCORE_COLUMNS, METRIC_COLUMNS, column_for, metric_name_for

SCHEMA_DEFAULT = "eval"
BRANCH_SCHEMA_DEFAULT = "branches"

# Postgres truncates identifiers at 63 bytes. The per-branch item view appends
# `__items` (7 characters), so a slug has to leave room for it -- otherwise two
# long branch names would be truncated to the SAME view name and silently share
# one window.
_MAX_IDENT = 63
_ITEMS_SUFFIX = "__items"
MAX_SLUG = _MAX_IDENT - len(_ITEMS_SUFFIX)


class StoreUnavailable(RuntimeError):
    """The store could not be opened: no driver, wrong credentials, server down.

    Deliberately distinct from a quality failure. A database that cannot be
    reached is a setup problem, and the CLI reports it as one rather than as a
    red gate.
    """


# ---------------------------------------------------------------------------
# branch name -> Postgres identifier
# ---------------------------------------------------------------------------

def slugify_branch(branch: str) -> str:
    """One git branch name as a readable, safe Postgres identifier.

    `feature/retrieval-k` -> `feature_retrieval_k`. Not reversible, so it can
    collide; `branch_slugs` resolves that across a whole set.
    """
    slug = re.sub(r"[^a-z0-9_]+", "_", (branch or "").lower())
    slug = re.sub(r"_{2,}", "_", slug).strip("_")
    if not slug:
        return "unknown"
    if slug[0].isdigit():
        # A PR ref is `42/merge`. An identifier starting with a digit is legal
        # only while quoted, and someone will eventually type it unquoted.
        slug = f"b_{slug}"
    return slug[:MAX_SLUG].rstrip("_") or "unknown"


def branch_slugs(branches) -> dict[str, str]:
    """{branch name: identifier}, collision-free and order-independent.

    Determinism matters: the view for a branch must keep the same name on every
    run, whatever order the rows came back in, or a saved query breaks.
    """
    out: dict[str, str] = {}
    taken: dict[str, str] = {}
    for branch in sorted({b for b in branches if b}):
        name = slugify_branch(branch)
        if taken.get(name, branch) != branch:
            # Two different branch names slugified the same way (`feature/x` and
            # `feature-x`). Disambiguate with a hash of the original so the
            # mapping stays stable, rather than with a counter that would shift
            # as branches come and go.
            digest = hashlib.sha256(branch.encode("utf-8")).hexdigest()[:6]
            name = f"{name[: MAX_SLUG - 7].rstrip('_')}_{digest}"
        taken[name] = branch
        out[branch] = name
    return out


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

# One source of truth: CREATE TABLE and the "add a column an older database is
# missing" migration are both generated from this, so the two cannot drift.
TABLES: dict[str, list[tuple[str, str]]] = {
    "runs": [
        ("id", "BIGSERIAL PRIMARY KEY"),
        ("started_at", "TIMESTAMPTZ NOT NULL"),
        ("finished_at", "TIMESTAMPTZ"),
        ("label", "TEXT"),
        ("git_sha", "TEXT"),
        ("git_branch", "TEXT"),
        ("dataset", "TEXT"),
        ("dataset_size", "INTEGER"),
        ("dataset_hash", "TEXT"),           # identity of the questions asked
        ("judge_config_hash", "TEXT"),      # identity of the grader
        ("target", "JSONB"),
        ("judge_model", "TEXT"),
        ("summary", "JSONB"),
        ("gate_passed", "BOOLEAN"),
        ("gate_status", "TEXT"),            # 'pass' | 'fail' | 'incomplete'
        ("gate_detail", "JSONB"),
        ("baseline_run_id", "BIGINT"),      # which run the paired test used
        ("hard_failure", "TEXT"),
    ],
    # One row per QUESTION: the evidence. No scores.
    "eval_data": [
        ("id", "BIGSERIAL PRIMARY KEY"),
        ("run_id", "BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE"),
        ("item_id", "TEXT NOT NULL"),
        ("kind", "TEXT"),                   # answerable | unanswerable
        ("question", "TEXT"),
        ("expected", "TEXT"),               # reference answer; no metric reads it
        ("answer", "TEXT"),
        ("contexts", "JSONB"),              # chunk text faithfulness was judged on
        ("sources", "JSONB"),               # the app's own citation strings
        ("retrieved", "JSONB"),             # ranked [{file, page, rank}]
        ("latency_s", "DOUBLE PRECISION"),
        ("input_tokens", "INTEGER"),
        ("output_tokens", "INTEGER"),
        ("cost_usd", "DOUBLE PRECISION"),
        ("error", "TEXT"),                  # the QUESTION failed (target down)
        ("reasons", "JSONB"),               # {metric: the judge's explanation}
        ("metric_errors", "JSONB"),         # {metric: why it was not scored}
    ],
    # One row per QUESTION: nothing but scores. A NULL means "not measured for
    # this question", which is NOT 0 and must survive into the paired test:
    # abstention_rate is NULL on every answerable question, not zero.
    "results": [
        ("id", "BIGSERIAL PRIMARY KEY"),
        ("run_id", "BIGINT NOT NULL REFERENCES runs(id) ON DELETE CASCADE"),
        ("item_id", "TEXT NOT NULL"),
        ("answered_rate", "DOUBLE PRECISION"),
        ("faithfulness", "DOUBLE PRECISION"),
        ("answer_relevance", "DOUBLE PRECISION"),
        ("hallucination_rate", "DOUBLE PRECISION"),
        ("abstention_rate", "DOUBLE PRECISION"),
        ("false_refusal_rate", "DOUBLE PRECISION"),
        ("retrieval_hit_rate", "DOUBLE PRECISION"),
        ("retrieval_recall", "DOUBLE PRECISION"),
        ("retrieval_mrr", "DOUBLE PRECISION"),
    ],
}

TABLE_CONSTRAINTS: dict[str, list[str]] = {}

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_eval_data_run ON eval_data(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_eval_data_item ON eval_data(run_id, item_id)",
    "CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_results_item ON results(run_id, item_id)",
    "CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at)",
    # Covers find_baseline exactly: the four comparability columns plus id.
    "CREATE INDEX IF NOT EXISTS idx_runs_baseline "
    "ON runs(git_branch, gate_status, dataset_hash, judge_config_hash, id)",
]




def create_table_sql(table: str) -> str:
    parts = [f"{name} {decl}" for name, decl in TABLES[table]]
    parts += TABLE_CONSTRAINTS.get(table, [])
    body = ",\n    ".join(parts)
    return f"CREATE TABLE IF NOT EXISTS {table} (\n    {body}\n)"


def migrate_sql() -> list[str]:
    """`ADD COLUMN IF NOT EXISTS` for every column that can be added later.

    Run history is the one thing here that cannot be regenerated -- the paired
    gate's baseline lives in exactly these rows -- so an upgrade must never
    require dropping it. Columns that are NOT NULL or part of a key are skipped:
    those cannot be bolted onto a populated table, and they have existed since
    the first version anyway.
    """
    out: list[str] = []
    for table, columns in TABLES.items():
        for name, decl in columns:
            upper = decl.upper()
            if "PRIMARY KEY" in upper or "NOT NULL" in upper:
                continue
            out.append(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {name} {decl}")
    return out


# One row per run, with the metrics whose names are fixed promoted to columns.
# Retrieval metrics carry `@k` in their name (`retrieval_hit_rate@5`), which
# changes with config, so they are not columns here -- read them from
# `summary_json->>'retrieval_hit_rate@5'` or from the `__items` view.
_SCOREBOARD_SQL = """
SELECT r.id                                          AS run_id,
       r.started_at,
       r.finished_at,
       r.label,
       r.git_sha,
       r.gate_status,
       r.dataset_size                                AS questions,
       r.baseline_run_id,
       (r.summary->>'faithfulness')::float8           AS faithfulness,
       (r.summary->>'answer_relevance')::float8       AS answer_relevance,
       (r.summary->>'hallucination_rate')::float8     AS hallucination_rate,
       (r.summary->>'abstention_rate')::float8        AS abstention_rate,
       (r.summary->>'false_refusal_rate')::float8     AS false_refusal_rate,
       (r.summary->>'retrieval_mrr')::float8          AS retrieval_mrr,
       (r.summary->>'latency_p50')::float8            AS latency_p50,
       (r.summary->>'latency_p95')::float8            AS latency_p95,
       (r.summary->>'judge_calls')::int               AS judge_calls,
       (r.summary->>'errors')::int                    AS errors,
       r.hard_failure,
       r.dataset_hash,
       r.judge_config_hash,
       r.judge_model,
       r.summary                                      AS summary_json
FROM {schema}.runs r
WHERE r.git_branch = {branch}
ORDER BY r.id DESC
"""

_ITEMS_SQL = """
SELECT r.id          AS run_id,
       r.started_at,
       r.git_sha,
       r.label,
       r.gate_status,
       d.item_id,
       d.kind,
       m.answered_rate,
       m.faithfulness,
       m.answer_relevance,
       m.hallucination_rate,
       m.abstention_rate,
       m.false_refusal_rate,
       m.retrieval_hit_rate,
       m.retrieval_recall,
       m.retrieval_mrr,
       d.latency_s,
       d.error,
       d.question,
       d.answer
FROM {schema}.runs r
JOIN {schema}.eval_data d ON d.run_id = r.id
LEFT JOIN {schema}.results m ON m.run_id = d.run_id AND m.item_id = d.item_id
WHERE r.git_branch = {branch}
ORDER BY r.id DESC, d.item_id
"""

_OVERVIEW_SQL = """
SELECT git_branch,
       count(*)                                              AS runs,
       count(*) FILTER (WHERE gate_status = 'pass')           AS passed,
       count(*) FILTER (WHERE gate_status = 'fail')           AS failed,
       count(*) FILTER (WHERE gate_status = 'incomplete')     AS incomplete,
       max(started_at)                                        AS last_run_at,
       max(id) FILTER (WHERE gate_status = 'pass')            AS last_passing_run_id
FROM {schema}.runs
WHERE git_branch IS NOT NULL
GROUP BY git_branch
ORDER BY max(started_at) DESC
"""


def _libpq_dirs() -> list[str]:
    """Directories that might hold a usable libpq, best first."""
    explicit = os.getenv("EVAL_LIBPQ_DIR")
    if explicit:
        return [explicit]
    if not sys.platform.startswith("win"):
        return []
    found = glob.glob(r"C:\Program Files\PostgreSQL\*\bin\libpq.dll")
    found += glob.glob(r"C:\Program Files (x86)\PostgreSQL\*\bin\libpq.dll")

    def version_key(path: str):
        try:
            return int(os.path.basename(os.path.dirname(os.path.dirname(path))))
        except ValueError:
            return -1

    return [os.path.dirname(p) for p in sorted(found, key=version_key, reverse=True)]


def _ensure_libpq() -> None:
    """Make an installed PostgreSQL's libpq loadable before psycopg looks for it.

    psycopg needs libpq. The `psycopg[binary]` wheel bundles its own copy, but
    that DLL is unsigned, and on Windows with Smart App Control enabled the
    loader refuses it ("An Application Control policy has blocked this file").
    psycopg then falls back to its pure-Python implementation, which loads a
    SYSTEM libpq via `ctypes.util.find_library` -- and PostgreSQL's own installer
    ships a signed one.

    So the fix is to point the loader at that file rather than to weaken a
    machine-wide security setting. `find_library` searches PATH on Windows,
    which is why PATH is what gets extended; `add_dll_directory` covers the
    loader's own search as well. Set EVAL_LIBPQ_DIR to override.

    A no-op everywhere else: on Linux and macOS the package manager's libpq is
    already on the loader path.
    """
    for directory in _libpq_dirs():
        if not os.path.isdir(directory):
            continue
        if directory not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
        add = getattr(os, "add_dll_directory", None)
        if add is not None:
            try:
                add(directory)
            except OSError:  # pragma: no cover - vanished between the two calls
                pass
        return


def _psycopg():
    _ensure_libpq()
    try:
        import psycopg
        from psycopg import sql
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
    except ImportError as exc:
        # ImportError, not ModuleNotFoundError: psycopg is often installed and
        # still unimportable because no libpq can be loaded -- which is a
        # different problem with a different fix, so both are named here.
        raise StoreUnavailable(
            f"the postgres backend needs psycopg and a loadable libpq: {exc}\n"
            '    install the driver:   pip install "psycopg[binary]"\n'
            "    if a security policy blocks the bundled libpq.dll, install\n"
            "    PostgreSQL itself (its libpq is signed) or point EVAL_LIBPQ_DIR\n"
            "    at the folder containing libpq\n"
            "    or pass --storage sqlite for an offline run"
        ) from exc
    return psycopg, sql, dict_row, Jsonb


def driver_available() -> bool:
    """Whether a Postgres connection could be attempted at all.

    Checks the import AND the libpq load, which are separate failure modes.
    """
    try:
        _psycopg()
    except StoreUnavailable:
        return False
    return True


class PostgresStore:
    """Run history in Postgres. Mirrors `store.Store` method for method."""

    backend = "postgres"

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        schema: str = SCHEMA_DEFAULT,
        branch_schema: str = BRANCH_SCHEMA_DEFAULT,
        **connect_params,
    ):
        self._psycopg, self._sql, dict_row, self._Jsonb = _psycopg()
        self.schema = schema
        self.branch_schema = branch_schema
        # Drop unset params so psycopg falls back to libpq's own defaults and to
        # ~/.pgpass, instead of sending an explicit `password=None`.
        params = {k: v for k, v in connect_params.items() if v not in (None, "")}
        self._params = params
        try:
            self._conn = self._psycopg.connect(
                conninfo or "", row_factory=dict_row, **params
            )
        except self._psycopg.Error as exc:
            raise StoreUnavailable(f"cannot connect to Postgres: {exc}".strip()) from exc
        try:
            self._init_schema()
        except self._psycopg.Error as exc:
            self._conn.close()
            raise StoreUnavailable(f"cannot set up the eval schema: {exc}".strip()) from exc

    # -- plumbing ----------------------------------------------------------

    def describe(self) -> str:
        host = self._params.get("host") or "localhost"
        dbname = self._params.get("dbname") or "?"
        return f"postgres {host}/{dbname} schema {self.schema}, views in {self.branch_schema}"

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    @contextmanager
    def _cursor(self):
        try:
            with self._conn.cursor() as cur:
                yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _ident(self, *names):
        return self._sql.Identifier(*names)

    def _has_column(self, cur, table: str, column: str) -> bool:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s AND column_name = %s",
            (self.schema, table, column),
        )
        return cur.fetchone() is not None

    def _has_table(self, cur, table: str) -> bool:
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = %s",
            (self.schema, table),
        )
        return cur.fetchone() is not None

    def _rename_before_schema(self, cur) -> None:
        """Renames that must run before CREATE TABLE / ADD COLUMN.

        An older database called the per-question table `results`. That name now
        belongs to the scores-only table, so the old one is renamed to
        `eval_data` first -- while `CREATE TABLE IF NOT EXISTS results` would
        still be a no-op and could not collide with it.

        A rename keeps every row. Run history cannot be regenerated: the paired
        gate's baseline IS these rows.
        """
        if self._has_table(cur, "results") and not self._has_table(cur, "eval_data"):
            if self._has_column(cur, "results", "question"):
                cur.execute("ALTER TABLE results RENAME TO eval_data")

        # `relevancy` was this metric's old column name.
        if self._has_table(cur, "eval_data"):
            if self._has_column(cur, "eval_data", "relevancy") and not self._has_column(
                cur, "eval_data", "answer_relevance"
            ):
                cur.execute(
                    "ALTER TABLE eval_data RENAME COLUMN relevancy TO answer_relevance"
                )

    def _fold_in_legacy_rows(self, cur) -> None:
        """Move any scores still sitting outside `results` into it, then clean up.

        Two shapes get folded in here:

        * score columns left on the per-question table (`faithfulness`,
          `relevancy`, the 0/1 `hallucinated` flag);
        * a whole second table, `item_results`, with one row per
          (question, metric) -- the same numbers again.

        Everything is copied BEFORE anything is dropped.
        """
        legacy = [
            c for c in _LEGACY_SCORE_COLUMNS if self._has_column(cur, "eval_data", c)
        ]
        has_item_results = self._has_table(cur, "item_results")
        if not legacy and not has_item_results:
            return

        picks = []
        for column in METRIC_COLUMNS:
            if self._has_column(cur, "eval_data", column):
                picks.append(f"d.{column}")
            elif column == "hallucination_rate" and self._has_column(
                cur, "eval_data", "hallucinated"
            ):
                # A 0/1 flag is the same fact as the rate the gate reads.
                picks.append("CASE WHEN d.hallucinated THEN 1.0 ELSE 0.0 END")
            else:
                picks.append("NULL")
        cur.execute(
            f"INSERT INTO results (run_id, item_id, {', '.join(METRIC_COLUMNS)}) "
            f"SELECT d.run_id, d.item_id, {', '.join(picks)} FROM eval_data d "
            f"WHERE NOT EXISTS (SELECT 1 FROM results r "
            f"                  WHERE r.run_id = d.run_id AND r.item_id = d.item_id)"
        )

        if has_item_results:
            for column in METRIC_COLUMNS:
                # `LIKE 'x@%'` catches the retrieval metrics, whose stored names
                # carried `@k`.
                cur.execute(
                    f"UPDATE results r SET {column} = ir.value "
                    f"FROM item_results ir "
                    f"WHERE ir.run_id = r.run_id AND ir.item_id = r.item_id "
                    f"  AND (ir.metric = %s OR ir.metric LIKE %s) "
                    f"  AND r.{column} IS NULL",
                    (column, f"{column}@%"),
                )
            cur.execute("DROP TABLE item_results")

        # `detail` was one JSON blob per question holding both of these.
        if self._has_column(cur, "eval_data", "detail"):
            for target, key in (("reasons", "reasons"), ("metric_errors", "metric_errors")):
                cur.execute(
                    f"UPDATE eval_data SET {target} = detail -> %s "
                    f"WHERE {target} IS NULL AND detail IS NOT NULL",
                    (key,),
                )
            cur.execute("ALTER TABLE eval_data DROP COLUMN detail")

        # Now that every score lives in `results`, strip the duplicates.
        for column in legacy:
            cur.execute(f"ALTER TABLE eval_data DROP COLUMN {column}")

    def _init_schema(self) -> None:
        sql = self._sql
        with self._cursor() as cur:
            for name in (self.schema, self.branch_schema):
                cur.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {s}").format(s=sql.Identifier(name))
                )
            # Everything below is written unqualified, against this search_path.
            cur.execute(
                sql.SQL("SET search_path TO {s}, public").format(s=sql.Identifier(self.schema))
            )
            self._rename_before_schema(cur)
            for table in TABLES:
                cur.execute(create_table_sql(table))
            for statement in migrate_sql():
                cur.execute(statement)
            self._fold_in_legacy_rows(cur)
            for statement in INDEXES:
                cur.execute(statement)
            overview = self._ident(self.schema, "branch_overview")
            cur.execute(sql.SQL("DROP VIEW IF EXISTS {view}").format(view=overview))
            cur.execute(
                sql.SQL("CREATE VIEW {view} AS " + _OVERVIEW_SQL).format(
                    view=overview, schema=sql.Identifier(self.schema)
                )
            )

    # -- writes ------------------------------------------------------------

    def start_run(self, **fields) -> int:
        fields["target"] = self._Jsonb(fields.get("target") or {})
        cols = ", ".join(fields)
        marks = ", ".join(["%s"] * len(fields))
        with self._cursor() as cur:
            cur.execute(
                f"INSERT INTO runs ({cols}) VALUES ({marks}) RETURNING id",
                tuple(fields.values()),
            )
            return int(cur.fetchone()["id"])

    def finish_run(
        self,
        run_id: int,
        finished_at: str,
        summary: dict,
        gate_passed: bool,
        gate_detail: list[dict],
        hard_failure: str | None = None,
        baseline_run_id: int | None = None,
    ) -> None:
        # One explicit status, because baseline selection filters on it and
        # "passed = false but hard_failure set" is not a quality verdict at all.
        status = "incomplete" if hard_failure else ("pass" if gate_passed else "fail")
        with self._cursor() as cur:
            cur.execute(
                "UPDATE runs SET finished_at=%s, summary=%s, gate_passed=%s, "
                "gate_status=%s, gate_detail=%s, hard_failure=%s, baseline_run_id=%s "
                "WHERE id=%s",
                (
                    finished_at,
                    self._Jsonb(summary),
                    bool(gate_passed),
                    status,
                    self._Jsonb(gate_detail),
                    hard_failure,
                    baseline_run_id,
                    run_id,
                ),
            )

    def add_result(self, run_id: int, result: dict, values: dict | None = None) -> None:
        """One question: its evidence into `eval_data`, its scores into `results`.

        Both rows go in one transaction. A half-written question -- evidence with
        no scores, or scores with no evidence -- would make the run look
        partially measured when it was not.

        `values` is {metric_name: value_or_None} for this one question, as
        `aggregate.values_by_item` produces. Metric names carrying `@k` are
        mapped onto their column, so changing `retrieval.k` never alters the
        schema.
        """
        data = {
            "run_id": run_id,
            "item_id": result.get("id"),
            "kind": result.get("kind"),
            "question": result.get("question"),
            "expected": result.get("expected_answer"),
            "answer": result.get("answer"),
            "contexts": self._Jsonb(result.get("contexts") or []),
            "sources": self._Jsonb(result.get("sources") or []),
            "retrieved": self._Jsonb(result.get("retrieved_sources") or []),
            "latency_s": result.get("latency_s"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cost_usd": result.get("cost_usd"),
            "error": result.get("error"),
            "reasons": self._Jsonb(result.get("reasons") or {}),
            "metric_errors": self._Jsonb(result.get("metric_errors") or {}),
        }
        scores = {"run_id": run_id, "item_id": result.get("id")}
        for column in METRIC_COLUMNS:
            scores[column] = None
        for metric, value in (values or {}).items():
            column = column_for(metric)
            if column in METRIC_COLUMNS:
                scores[column] = value

        with self._cursor() as cur:
            for table, row in (("eval_data", data), ("results", scores)):
                cols = ", ".join(row)
                marks = ", ".join(["%s"] * len(row))
                cur.execute(
                    f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values())
                )

    # -- reads -------------------------------------------------------------

    def _retrieval_k(self, cur, run_id: int) -> int | None:
        """The k this run used, from its stored summary.

        Needed to turn the `retrieval_hit_rate` column back into the metric name
        `retrieval_hit_rate@5` that the gate and the paired test look up.
        """
        cur.execute("SELECT summary FROM runs WHERE id=%s", (run_id,))
        row = cur.fetchone()
        summary = (row or {}).get("summary") or {}
        return summary.get("retrieval_k") if isinstance(summary, dict) else None

    def item_values(self, run_id: int) -> dict[str, dict]:
        """Read back {metric: {item_id: value_or_None}} for one run.

        This is what the paired comparison joins on, so the shape has to match
        `aggregate.item_metric_values` exactly -- including that a metric with no
        measurement anywhere is ABSENT rather than present and all-None. A metric
        of all-Nones would show up in the report as a row with no pairs, which
        reads like a fault rather than like "this run did not measure that".
        """
        columns = ", ".join(METRIC_COLUMNS)
        with self._cursor() as cur:
            retrieval_k = self._retrieval_k(cur, run_id)
            cur.execute(
                f"SELECT item_id, {columns} FROM results WHERE run_id=%s", (run_id,)
            )
            rows = cur.fetchall()
        out: dict[str, dict] = {}
        for column in METRIC_COLUMNS:
            per_item = {row["item_id"]: row[column] for row in rows}
            if any(value is not None for value in per_item.values()):
                out[metric_name_for(column, retrieval_k)] = per_item
        return out

    def find_baseline(
        self,
        branch: str,
        dataset_hash: str | None,
        judge_config_hash: str | None,
        exclude_run_id: int | None = None,
    ) -> dict | None:
        """The most recent PASSING run on `branch` that is comparable to this one.

        Comparable means the same questions and the same grader. Without both, a
        paired difference measures a change in the benchmark rather than in the
        app -- so a mismatch yields no baseline and the caller skips the paired
        test instead of reporting a regression it cannot substantiate.

        `IS NOT DISTINCT FROM` rather than `=`: a NULL hash must match a NULL
        hash, which plain equality in SQL never does.
        """
        sql = (
            "SELECT * FROM runs WHERE git_branch=%s AND gate_status='pass' "
            "AND dataset_hash IS NOT DISTINCT FROM %s "
            "AND judge_config_hash IS NOT DISTINCT FROM %s "
            "AND finished_at IS NOT NULL"
        )
        params: list = [branch, dataset_hash, judge_config_hash]
        if exclude_run_id is not None:
            sql += " AND id <> %s"
            params.append(exclude_run_id)
        sql += " ORDER BY id DESC LIMIT 1"
        with self._cursor() as cur:
            cur.execute(sql, tuple(params))
            row = cur.fetchone()
        return None if row is None else _decode_run(dict(row))

    def history(self, limit: int = 50) -> list[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM runs ORDER BY id DESC LIMIT %s", (limit,))
            rows = cur.fetchall()
        return [_decode_run(dict(row)) for row in rows]

    def results_for(self, run_id: int) -> list[dict]:
        """Every question of one run: its evidence and its scores, joined.

        The two tables exist so that a query for scores alone does not have to
        drag kilobytes of answer text with it -- but a human debugging a bad
        score wants both, so this hands over the join.
        """
        columns = ", ".join(f"m.{c}" for c in METRIC_COLUMNS)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT d.*, {columns} FROM eval_data d "
                f"LEFT JOIN results m ON m.run_id = d.run_id AND m.item_id = d.item_id "
                f"WHERE d.run_id=%s ORDER BY d.id",
                (run_id,),
            )
            return [dict(row) for row in cur.fetchall()]

    def eval_data_for(self, run_id: int) -> list[dict]:
        """The evidence only: question, answer, contexts, retrieved, cost."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM eval_data WHERE run_id=%s ORDER BY id", (run_id,))
            return [dict(row) for row in cur.fetchall()]

    def metrics_for(self, run_id: int) -> list[dict]:
        """The scores only, one row per question."""
        with self._cursor() as cur:
            cur.execute("SELECT * FROM results WHERE run_id=%s ORDER BY id", (run_id,))
            return [dict(row) for row in cur.fetchall()]

    def branches(self) -> list[str]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT DISTINCT git_branch FROM runs "
                "WHERE git_branch IS NOT NULL AND git_branch <> ''"
            )
            return sorted(row["git_branch"] for row in cur.fetchall())

    # -- per-branch views --------------------------------------------------

    def refresh_branch_views(self) -> dict[str, str]:
        """Give every branch that has runs its own pair of views.

        DROP then CREATE rather than CREATE OR REPLACE: replacing a view fails
        outright if its column list changed, which happens whenever a metric is
        added. A view holds no data, so dropping it costs nothing.
        """
        sql = self._sql
        mapping = branch_slugs(self.branches())
        wanted: set[str] = set()
        for slug in mapping.values():
            wanted |= {slug, f"{slug}{_ITEMS_SUFFIX}"}

        with self._cursor() as cur:
            for branch, slug in mapping.items():
                for name, body in (
                    (slug, _SCOREBOARD_SQL),
                    (f"{slug}{_ITEMS_SUFFIX}", _ITEMS_SQL),
                ):
                    target = self._ident(self.branch_schema, name)
                    cur.execute(sql.SQL("DROP VIEW IF EXISTS {view}").format(view=target))
                    cur.execute(
                        sql.SQL("CREATE VIEW {view} AS " + body).format(
                            view=target,
                            schema=sql.Identifier(self.schema),
                            branch=sql.Literal(branch),
                        )
                    )
            # Drop windows onto branches that no longer have any runs, so the
            # schema stays a list of what actually exists.
            cur.execute(
                "SELECT table_name FROM information_schema.views WHERE table_schema = %s",
                (self.branch_schema,),
            )
            for stale in sorted({r["table_name"] for r in cur.fetchall()} - wanted):
                cur.execute(
                    sql.SQL("DROP VIEW IF EXISTS {view}").format(
                        view=self._ident(self.branch_schema, stale)
                    )
                )
        return mapping


def _decode_run(rec: dict) -> dict:
    """Make a Postgres row look exactly like a SQLite one to callers.

    JSONB already arrives as dicts, so only the timestamps need flattening: the
    CLI and the reports treat `started_at` as an ISO string and slice it.
    """
    for key in ("started_at", "finished_at"):
        value = rec.get(key)
        if value is not None and hasattr(value, "isoformat"):
            rec[key] = value.isoformat(timespec="seconds")
    return rec
