"""Choose and open the run-history store.

Two backends, one API:

* **postgres** -- the real one. A database you can open in pgAdmin, point a
  dashboard at, and keep across machines. Each branch gets its own pair of views
  under the `branches` schema.
* **sqlite** -- a single file, no server. What the unit suite and a runner with
  nothing installed use, and the fallback in CI when no database URL is set.

Credentials never live in `eval_config.yaml`. The config names the ENV VAR that
holds the password (`password_env`), and that is read here. A full connection
string in `EVAL_DATABASE_URL` overrides every individual setting.

A note on that, because it bites: a URL-style DSN has to percent-encode the
password, so `abi@123` must be written `abi%40123` inside a URL. The host, port,
dbname and user path below passes keyword parameters to libpq instead, where no
encoding applies -- which is why that is the documented route and the URL is the
escape hatch.
"""
import os

from .config import resolve
from .store import Store
from .store_pg import StoreUnavailable  # re-exported: callers catch one type

BACKENDS = ("postgres", "sqlite")

__all__ = ["BACKENDS", "StoreUnavailable", "open_store", "postgres_params", "resolve_backend"]


def resolve_backend(cfg: dict, override: str | None = None) -> str:
    """Explicit flag, then environment, then config. First one wins."""
    backend = (
        override
        or os.getenv("EVAL_STORAGE_BACKEND")
        or ((cfg.get("storage") or {}).get("backend"))
        or "sqlite"
    ).strip().lower()
    if backend not in BACKENDS:
        raise StoreUnavailable(
            f"unknown storage backend {backend!r}; expected one of {', '.join(BACKENDS)}"
        )
    return backend


def postgres_params(cfg: dict) -> dict:
    """Connection settings, with env vars overriding the config file.

    Returns the kwargs `PostgresStore` takes, so the CLI can print them (minus
    the password) before trying to connect.
    """
    pg = dict(((cfg.get("storage") or {}).get("postgres")) or {})
    password_env = pg.get("password_env") or "EVAL_PG_PASSWORD"

    params = {
        "conninfo": os.getenv("EVAL_DATABASE_URL") or None,
        "host": os.getenv("EVAL_PG_HOST") or pg.get("host") or "localhost",
        "port": os.getenv("EVAL_PG_PORT") or pg.get("port") or 5432,
        "dbname": os.getenv("EVAL_PG_DBNAME") or pg.get("dbname") or "llm_eval_pipeline",
        "user": os.getenv("EVAL_PG_USER") or pg.get("user") or "postgres",
        "password": os.getenv(password_env) or None,
        # Bound the wait. Without it libpq can block for minutes on a host that
        # is routed but silent, and a hung eval run is worse than a failed one:
        # it holds the CI job open and reports nothing at all.
        "connect_timeout": int(pg.get("connect_timeout") or 10),
        "schema": pg.get("schema") or "eval",
        "branch_schema": pg.get("branch_schema") or "branches",
    }
    if params["conninfo"]:
        # A full DSN carries host, port, database and credentials itself. Keeping
        # the individual settings as well would silently override parts of it.
        for key in ("host", "port", "dbname", "user", "password"):
            params[key] = None
    return params


def describe_params(params: dict) -> str:
    """A printable summary that never contains the password."""
    if params.get("conninfo"):
        return "postgres via $EVAL_DATABASE_URL"
    return (
        f"postgres {params['user']}@{params['host']}:{params['port']}/{params['dbname']} "
        f"(schema {params['schema']}, branch views in {params['branch_schema']})"
    )


# The database every PostgreSQL server has, used only as a doorway for
# `CREATE DATABASE` -- which cannot be run from inside the database it creates.
MAINTENANCE_DB = "postgres"


def _is_missing_database(exc: Exception) -> bool:
    """Whether a connection failed specifically because the database is absent.

    Matched on libpq's message because the error arrives before a connection
    exists, so there is no SQLSTATE to read.
    """
    return "does not exist" in str(exc).lower()


def _similar_database_names(conn, target: str) -> list[str]:
    """Existing databases whose names differ from `target` only in punctuation.

    This is the typo net. A database called `" llm_eval_pipeline"` -- a real
    pgAdmin slip, leading space and all -- normalises to the same thing as
    `llm_eval_pipeline`, so creating the second one silently would leave two
    databases and no clue which holds the rows.
    """
    rows = conn.execute(
        "SELECT datname FROM pg_database WHERE datname <> %s AND "
        "regexp_replace(lower(datname), '[^a-z0-9]', '', 'g') = "
        "regexp_replace(lower(%s), '[^a-z0-9]', '', 'g')",
        (target, target),
    ).fetchall()
    return sorted(r["datname"] for r in rows)


def create_database(params: dict) -> list[str]:
    """`CREATE DATABASE` the configured database. Returns warnings to show.

    Connects to the maintenance database to do it, and in autocommit, because
    CREATE DATABASE cannot run inside a transaction.
    """
    from .store_pg import _psycopg

    psycopg, sql, dict_row, _ = _psycopg()
    target = params["dbname"]
    admin = {
        key: params[key]
        for key in ("host", "port", "user", "password", "connect_timeout")
        if params.get(key) is not None
    }
    admin["dbname"] = MAINTENANCE_DB

    try:
        conn = psycopg.connect(autocommit=True, row_factory=dict_row, **admin)
    except psycopg.Error as exc:
        raise StoreUnavailable(
            f"database {target!r} does not exist, and connecting to the "
            f"{MAINTENANCE_DB!r} database to create it also failed:\n    {exc}\n"
            f"  create it by hand instead:  CREATE DATABASE \"{target}\";"
        ) from exc

    warnings: list[str] = []
    try:
        similar = _similar_database_names(conn, target)
        try:
            conn.execute(sql.SQL("CREATE DATABASE {db}").format(db=sql.Identifier(target)))
        except psycopg.errors.DuplicateDatabase:
            # Another process got there first between the failed connect and
            # now. Nothing to warn about: the database exists, which is the goal.
            return []
        except psycopg.errors.InsufficientPrivilege as exc:
            raise StoreUnavailable(
                f"database {target!r} does not exist and user "
                f"{params.get('user')!r} may not create one:\n    {exc}\n"
                f"  ask for CREATEDB, or have someone run:  "
                f'CREATE DATABASE "{target}";'
            ) from exc
        except psycopg.Error as exc:
            raise StoreUnavailable(
                f"could not create database {target!r}:\n    {exc}"
            ) from exc

        warnings.append(f"created database {target!r} (it did not exist)")
        if similar:
            names = ", ".join(repr(n) for n in similar)
            warnings.append(
                f"WARNING: these databases already existed with almost the same "
                f"name: {names}"
            )
            warnings.append(
                f"         If one of those is the one you meant, you now have two "
                f"and the rows will go to {target!r}. Check "
                f"storage.postgres.dbname in eval_config.yaml."
            )
    finally:
        conn.close()
    return warnings


def open_store(
    cfg: dict,
    backend: str | None = None,
    create_db: bool | None = None,
    notify=None,
):
    """Open the configured store, or raise `StoreUnavailable` with the reason.

    If the database itself is missing it is created, unless that is switched off.
    Everything inside it -- schemas, tables, indexes, views, and columns added by
    a later version -- is created on every open regardless, so there is never any
    DDL to write by hand.

    There is deliberately NO automatic fallback from postgres to sqlite. Falling
    back silently would mean a run reports success while the rows the user went
    looking for are in a file they did not ask for -- and, worse, the paired
    gate would compare against a baseline from the wrong database.
    """
    notify = notify or (lambda message: None)
    which = resolve_backend(cfg, backend)
    if which == "sqlite":
        return Store(sqlite_path(cfg))

    from .store_pg import PostgresStore  # imported late: psycopg is optional

    params = postgres_params(cfg)
    pg_cfg = ((cfg.get("storage") or {}).get("postgres")) or {}
    may_create = pg_cfg.get("create_database", True) if create_db is None else create_db

    try:
        return PostgresStore(**params)
    except StoreUnavailable as exc:
        if not _is_missing_database(exc):
            raise StoreUnavailable(
                f"{exc}\n  tried: {describe_params(params)}\n"
                f"  check the server is running, or pass --storage sqlite for an "
                f"offline run"
            ) from exc

        if params.get("conninfo"):
            # The database name lives inside the DSN, and a hosted Postgres
            # normally provisions the database for you anyway.
            raise StoreUnavailable(
                f"{exc}\n  the connection came from $EVAL_DATABASE_URL, so the "
                f"database is not created automatically.\n"
                f"  create it on the server, or unset EVAL_DATABASE_URL to use "
                f"storage.postgres in the config."
            ) from exc

        if not may_create:
            raise StoreUnavailable(
                f"{exc}\n  create it with:  CREATE DATABASE "
                f'"{params["dbname"]}";\n'
                f"  or re-run with --create-database to have this do it for you"
            ) from exc

        for message in create_database(params):
            notify(message)
        return PostgresStore(**params)


def sqlite_path(cfg: dict):
    storage = cfg.get("storage") or {}
    return resolve(
        storage.get("sqlite_path") or ".llmeval/history.sqlite", cfg
    )


def env_file_hint(cfg: dict) -> str:
    """Where to put the password, next to the config that asked for it."""
    from pathlib import Path

    return (
        f"put the password in {Path(cfg.get('_dir') or '.') / '.env'} "
        f"as EVAL_PG_PASSWORD=..."
    )
