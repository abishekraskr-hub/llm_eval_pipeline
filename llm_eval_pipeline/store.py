"""Run history in SQLite.

SQLite because a file needs no service to run in CI and can be committed as an
artifact. The schema is deliberately Postgres-shaped (JSON columns, integer keys)
so swapping the backend later is a connection change, not a redesign.

Three tables, keyed `(run_id, item_id)`:

* `runs`      -- one row per evaluation, with the aggregate summary and verdict
* `eval_data` -- one row per QUESTION: what was asked, what came back, what it
                 cost, and the judge's reasoning
* `results`   -- one row per QUESTION: nothing but the metric scores, one column
                 each

Each score is written in exactly ONE place: `results`. Nothing else holds a
score, so no two copies of a number can disagree -- which is what happened when
a second long-format table held them all again.

The split keeps `results` narrow and numeric. `answer` and `contexts` run to
kilobytes per row, so a `SELECT *` over the scores would otherwise drag the
whole corpus along with it.

Note on the retrieval columns: the metric is NAMED `retrieval_hit_rate@5`, but
the COLUMN is `retrieval_hit_rate`. `k` is a property of the run, recorded once
in `runs.summary`, not of each row -- baking it into a column name would mean an
ALTER TABLE every time `retrieval.k` changed.
"""
import json
import sqlite3
from contextlib import closing
from pathlib import Path

# Every per-question metric, in report order. Adding a metric means adding it
# here and to the SCHEMA below; the migration then adds the column to an
# existing database on the next open.
METRIC_COLUMNS = (
    "answered_rate",
    "faithfulness",
    "answer_relevance",
    "hallucination_rate",
    "abstention_rate",
    "false_refusal_rate",
    "retrieval_hit_rate",
    "retrieval_recall",
    "retrieval_mrr",
)

# The two metrics whose NAME carries `@k` while their column does not.
_K_SUFFIXED = ("retrieval_hit_rate", "retrieval_recall")

# Score-carrying columns an older database kept on its per-question table. They
# are moved into `results` and then dropped, so a score has one home again.
_LEGACY_SCORE_COLUMNS = METRIC_COLUMNS + ("relevancy", "hallucinated")


def column_for(metric: str) -> str:
    """Metric name to column name: `retrieval_hit_rate@5` -> `retrieval_hit_rate`."""
    return metric.split("@", 1)[0]


def metric_name_for(column: str, retrieval_k: int | None) -> str:
    """Column name to metric name: `retrieval_hit_rate` + k=5 -> `...@5`.

    The gate and the paired test look metrics up by name, and a hit rate at 5 is
    not the same measurement as one at 20 -- so the name has to carry k even
    though the column does not.
    """
    if retrieval_k and column in _K_SUFFIXED:
        return f"{column}@{retrieval_k}"
    return column


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at        TEXT    NOT NULL,
    finished_at       TEXT,
    label             TEXT,
    git_sha           TEXT,
    git_branch        TEXT,
    dataset           TEXT,
    dataset_size      INTEGER,
    dataset_hash      TEXT,     -- identity of the questions asked
    judge_config_hash TEXT,     -- identity of the grader
    target            TEXT,     -- JSON: the target's describe()
    judge_model       TEXT,
    summary           TEXT,     -- JSON: RunSummary
    gate_passed       INTEGER,
    gate_status       TEXT,     -- 'pass' | 'fail' | 'incomplete'
    gate_detail       TEXT,     -- JSON: per-threshold checks
    baseline_run_id   INTEGER,  -- which run the paired test compared against
    hard_failure      TEXT
);

-- One row per QUESTION: the evidence. No scores.
CREATE TABLE IF NOT EXISTS eval_data (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    item_id       TEXT    NOT NULL,
    kind          TEXT,     -- answerable | unanswerable
    question      TEXT,
    expected      TEXT,     -- reference answer; no metric reads it yet
    answer        TEXT,
    contexts      TEXT,     -- JSON: the chunk text faithfulness was judged on
    sources       TEXT,     -- JSON: the app's own citation strings
    retrieved     TEXT,     -- JSON: ranked [{file, page, rank}]
    latency_s     REAL,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    cost_usd      REAL,
    error         TEXT,     -- the QUESTION failed outright (target down)
    reasons       TEXT,     -- JSON {metric: the judge's explanation}
    metric_errors TEXT      -- JSON {metric: why it could not be scored}
);

-- One row per QUESTION: nothing but scores. A NULL means "not measured for this
-- question", which is NOT 0 and must survive into the paired test --
-- abstention_rate is NULL on every answerable question, not zero.
CREATE TABLE IF NOT EXISTS results (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    item_id       TEXT    NOT NULL,

    answered_rate      REAL,
    faithfulness       REAL,
    answer_relevance   REAL,
    hallucination_rate REAL,
    abstention_rate    REAL,
    false_refusal_rate REAL,
    retrieval_hit_rate REAL,
    retrieval_recall   REAL,
    retrieval_mrr      REAL
);

"""

# Applied AFTER the column migration below, not with the tables: an index over
# `gate_status` cannot be created on a database from before that column existed.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_eval_data_run ON eval_data(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_data_item ON eval_data(run_id, item_id);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_item ON results(run_id, item_id);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_runs_baseline
    ON runs(git_branch, gate_status, dataset_hash, judge_config_hash, id);
"""

_TABLES = ("runs", "eval_data", "results")

# Columns the old wide `results` carried that are now duplicates: the scores are
# metric rows in `results`, and `detail` held the judge reasons and per-metric
# errors that are now `results.reason` and `results.error`.
_LEGACY_DUPLICATE_COLUMNS = ("faithfulness", "relevancy", "hallucinated", "detail")


class Store:
    backend = "sqlite"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            self._rename_before_schema(conn)
            conn.executescript(SCHEMA)
            self._migrate(conn)
            self._fold_in_legacy_rows(conn)
            conn.executescript(INDEXES)
            conn.commit()

    def describe(self) -> str:
        return f"sqlite {self.path}"

    def close(self) -> None:
        """Nothing to close: a connection is opened and shut per call."""

    def refresh_branch_views(self) -> dict[str, str]:
        """No-op. SQLite has no schemas, so there is nowhere to put per-branch
        views; `SELECT ... WHERE git_branch = ?` is the equivalent here. The
        method exists so callers never have to ask which backend they hold."""
        return {}

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _desired_columns() -> dict[str, list[tuple[str, str]]]:
        """What each table should look like, read back from SCHEMA itself.

        Deriving this instead of hand-listing "columns added since v1" removes a
        trap: a hand-maintained list has to be updated in lockstep with SCHEMA, and
        forgetting means an old database quietly fails at runtime instead of being
        migrated. Here the two cannot drift, because there is only one source.
        """
        probe = sqlite3.connect(":memory:")
        try:
            probe.executescript(SCHEMA)
            desired: dict[str, list[tuple[str, str]]] = {}
            for table in _TABLES:
                rows = probe.execute(f"PRAGMA table_info({table})").fetchall()
                # name, declared type, notnull, pk -- skip anything ALTER TABLE
                # cannot add to an existing table.
                desired[table] = [
                    (row[1], row[2]) for row in rows if not row[3] and not row[5]
                ]
            return desired
        finally:
            probe.close()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add columns a database created by an older version is missing.

        Run history is the one thing here that cannot be regenerated, so an upgrade
        must never require deleting it -- the baseline for the paired gate lives in
        exactly these rows.
        """
        for table, columns in self._desired_columns().items():
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if not existing:
                continue  # table does not exist at all; SCHEMA already created it
            for column, decl in columns:
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


    @staticmethod
    def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone() is not None

    def _rename_before_schema(self, conn: sqlite3.Connection) -> None:
        """Renames that must run before CREATE TABLE / ADD COLUMN.

        An older database called the per-question table `results`. That name now
        belongs to the scores-only table, so the old one is renamed to
        `eval_data` first -- while `CREATE TABLE IF NOT EXISTS results` would
        still be a no-op and could not collide with it.

        A rename keeps every row. Run history cannot be regenerated: the paired
        gate's baseline IS these rows.
        """
        if self._table_exists(conn, "results") and not self._table_exists(conn, "eval_data"):
            if "question" in self._columns(conn, "results"):
                conn.execute("ALTER TABLE results RENAME TO eval_data")

        # `relevancy` was this metric's old column name.
        if self._table_exists(conn, "eval_data"):
            columns = self._columns(conn, "eval_data")
            if "relevancy" in columns and "answer_relevance" not in columns:
                conn.execute("ALTER TABLE eval_data RENAME COLUMN relevancy TO answer_relevance")
        conn.commit()

    def _fold_in_legacy_rows(self, conn: sqlite3.Connection) -> None:
        """Move any scores still sitting outside `results` into it, then clean up.

        Two shapes get folded in here:

        * score columns left on the per-question table (`faithfulness`,
          `relevancy`, the 0/1 `hallucinated` flag);
        * a whole second table, `item_results`, that held one row per
          (question, metric) -- the same numbers again.

        Everything is copied BEFORE anything is dropped.
        """
        data_columns = self._columns(conn, "eval_data")
        legacy = [c for c in _LEGACY_SCORE_COLUMNS if c in data_columns]
        has_item_results = self._table_exists(conn, "item_results")
        if not legacy and not has_item_results:
            return

        # One `results` row per question that does not have one yet, carrying
        # whatever scores the old per-question table still holds.
        picks = []
        for column in METRIC_COLUMNS:
            if column in data_columns:
                picks.append(f"d.{column}")
            elif column == "hallucination_rate" and "hallucinated" in data_columns:
                # A 0/1 flag is the same fact as the rate the gate reads.
                picks.append("CASE WHEN d.hallucinated THEN 1.0 ELSE 0.0 END")
            else:
                picks.append("NULL")
        conn.execute(
            f"INSERT INTO results (run_id, item_id, {', '.join(METRIC_COLUMNS)}) "
            f"SELECT d.run_id, d.item_id, {', '.join(picks)} FROM eval_data d "
            f"WHERE NOT EXISTS (SELECT 1 FROM results r "
            f"                  WHERE r.run_id = d.run_id AND r.item_id = d.item_id)"
        )

        if has_item_results:
            for column in METRIC_COLUMNS:
                # `OR metric LIKE 'x@%'` catches the retrieval metrics, whose
                # stored names carried `@k`.
                conn.execute(
                    f"UPDATE results SET {column} = ("
                    f"  SELECT ir.value FROM item_results ir"
                    f"   WHERE ir.run_id = results.run_id AND ir.item_id = results.item_id"
                    f"     AND (ir.metric = ? OR ir.metric LIKE ?)"
                    f"   LIMIT 1) "
                    f"WHERE {column} IS NULL",
                    (column, f"{column}@%"),
                )
            conn.execute("DROP TABLE item_results")

        # `detail` was one JSON blob per question holding both of these.
        if "detail" in data_columns:
            for target, key in (("reasons", "$.reasons"), ("metric_errors", "$.metric_errors")):
                conn.execute(
                    f"UPDATE eval_data SET {target} = json_extract(detail, '{key}') "
                    f"WHERE {target} IS NULL AND detail IS NOT NULL"
                )
            conn.execute("ALTER TABLE eval_data DROP COLUMN detail")

        # Now that every score lives in `results`, strip the duplicates.
        for column in legacy:
            conn.execute(f"ALTER TABLE eval_data DROP COLUMN {column}")
        conn.commit()

    def start_run(self, **fields) -> int:
        fields["target"] = json.dumps(fields.get("target") or {})
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        with closing(self._connect()) as conn:
            cur = conn.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", tuple(fields.values()))
            conn.commit()
            return int(cur.lastrowid)

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
        # "passed == 0 but hard_failure set" is not a quality verdict at all.
        status = "incomplete" if hard_failure else ("pass" if gate_passed else "fail")
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE runs SET finished_at=?, summary=?, gate_passed=?, gate_status=?, "
                "gate_detail=?, hard_failure=?, baseline_run_id=? WHERE id=?",
                (
                    finished_at,
                    json.dumps(summary),
                    1 if gate_passed else 0,
                    status,
                    json.dumps(gate_detail),
                    hard_failure,
                    baseline_run_id,
                    run_id,
                ),
            )
            conn.commit()

    def add_result(self, run_id: int, result: dict, values: dict | None = None) -> None:
        """One question: its evidence into `eval_data`, its scores into `results`.

        Both rows are written in a single transaction. A half-written question --
        evidence with no scores, or scores with no evidence -- would make the run
        look partially measured when it was not.

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
            "contexts": json.dumps(result.get("contexts") or []),
            "sources": json.dumps(result.get("sources") or []),
            "retrieved": json.dumps(result.get("retrieved_sources") or []),
            "latency_s": result.get("latency_s"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cost_usd": result.get("cost_usd"),
            "error": result.get("error"),
            "reasons": json.dumps(result.get("reasons") or {}),
            "metric_errors": json.dumps(result.get("metric_errors") or {}),
        }
        scores = {"run_id": run_id, "item_id": result.get("id")}
        for column in METRIC_COLUMNS:
            scores[column] = None
        for metric, value in (values or {}).items():
            column = column_for(metric)
            if column in METRIC_COLUMNS:
                scores[column] = value

        with closing(self._connect()) as conn:
            for table, row in (("eval_data", data), ("results", scores)):
                cols = ", ".join(row)
                marks = ", ".join("?" for _ in row)
                conn.execute(
                    f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(row.values())
                )
            conn.commit()

    def _retrieval_k(self, conn: sqlite3.Connection, run_id: int) -> int | None:
        """The k this run used, from its stored summary.

        Needed to turn the `retrieval_hit_rate` column back into the metric name
        `retrieval_hit_rate@5` that the gate and the paired test look up.
        """
        row = conn.execute("SELECT summary FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None or not row["summary"]:
            return None
        try:
            return (json.loads(row["summary"]) or {}).get("retrieval_k")
        except (json.JSONDecodeError, TypeError):
            return None

    def item_values(self, run_id: int) -> dict[str, dict]:
        """Read back {metric: {item_id: value_or_None}} for one run.

        This is what the paired comparison joins on, so the shape has to match
        `aggregate.item_metric_values` exactly -- including that a metric with no
        measurement anywhere is ABSENT rather than present and all-None. A metric
        of all-Nones would show up in the report as a row with no pairs, which
        reads like a fault rather than like "this run did not measure that".
        """
        columns = ", ".join(METRIC_COLUMNS)
        with closing(self._connect()) as conn:
            retrieval_k = self._retrieval_k(conn, run_id)
            rows = conn.execute(
                f"SELECT item_id, {columns} FROM results WHERE run_id=?", (run_id,)
            ).fetchall()
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

        Comparable means same questions and same grader. Without both, a paired
        difference measures the change in the benchmark rather than in the app --
        so a mismatch yields no baseline, and the caller skips the paired test
        instead of reporting a regression it cannot substantiate.
        """
        sql = (
            "SELECT * FROM runs WHERE git_branch=? AND gate_status='pass' "
            "AND dataset_hash IS ? AND judge_config_hash IS ? "
            "AND finished_at IS NOT NULL"
        )
        params: list = [branch, dataset_hash, judge_config_hash]
        if exclude_run_id is not None:
            sql += " AND id <> ?"
            params.append(exclude_run_id)
        sql += " ORDER BY id DESC LIMIT 1"
        with closing(self._connect()) as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return None
        return _decode_run(dict(row))

    def history(self, limit: int = 50) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_decode_run(dict(row)) for row in rows]

    def results_for(self, run_id: int) -> list[dict]:
        """Every question of one run: its evidence and its scores, joined.

        The two tables exist so that a query for scores alone does not have to
        drag kilobytes of answer text with it -- but a human debugging a bad
        score wants both, so this hands over the join.
        """
        columns = ", ".join(f"r.{c}" for c in METRIC_COLUMNS)
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT d.*, {columns} FROM eval_data d "
                f"LEFT JOIN results r ON r.run_id = d.run_id AND r.item_id = d.item_id "
                f"WHERE d.run_id=? ORDER BY d.id",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def eval_data_for(self, run_id: int) -> list[dict]:
        """The evidence only: question, answer, contexts, retrieved, cost."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM eval_data WHERE run_id=? ORDER BY id", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def metrics_for(self, run_id: int) -> list[dict]:
        """The scores only, one row per question."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT * FROM results WHERE run_id=? ORDER BY id", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]


def _decode_run(rec: dict) -> dict:
    for key in ("summary", "gate_detail", "target"):
        if rec.get(key):
            try:
                rec[key] = json.loads(rec[key])
            except json.JSONDecodeError:
                pass
    return rec
