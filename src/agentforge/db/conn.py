"""Postgres access: one pool, explicit schema application, verified extensions.

The whole point of ParadeDB here is that `pg_search` (Tantivy-backed Okapi BM25)
and `pgvector` are the same database, so a chunk is lexically and semantically
searchable in the same transaction. There is no second store and no sync path,
so there is no window in which the two indexes disagree.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from agentforge.errors import IndexingError
from agentforge.settings import settings

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

REQUIRED_EXTENSIONS = ("vector", "pg_search")


@lru_cache(maxsize=1)
def pool() -> ConnectionPool:
    return ConnectionPool(
        settings().pg_dsn,
        min_size=1,
        max_size=8,
        kwargs={"row_factory": dict_row},
        open=True,
    )


def apply_schema() -> None:
    """Idempotent. Safe to call on every boot; CREATE ... IF NOT EXISTS throughout."""
    sql = SCHEMA_PATH.read_text()
    with pool().connection() as conn:
        conn.execute(sql)  # type: ignore[arg-type]


def require_extensions() -> dict[str, str]:
    """Verify both extensions are installed, and return their versions.

    Raises rather than degrading: a missing pg_search would otherwise silently
    push the lexical arm onto `ts_rank`, which is not BM25 and would make the
    reported hybrid numbers describe a different system than the one claimed.
    """
    with pool().connection() as conn:
        rows = conn.execute(
            "SELECT extname, extversion FROM pg_extension WHERE extname = ANY(%s)",
            (list(REQUIRED_EXTENSIONS),),
        ).fetchall()
    found = {r["extname"]: r["extversion"] for r in rows}
    missing = [e for e in REQUIRED_EXTENSIONS if e not in found]
    if missing:
        raise IndexingError(
            f"missing Postgres extension(s): {', '.join(missing)}. "
            "The db service must be the paradedb/paradedb image, which ships both."
        )
    return found


def server_info() -> dict[str, str]:
    """Recorded alongside runs: the index implementation is part of the result."""
    with pool().connection() as conn:
        version = conn.execute("SELECT version() AS v").fetchone()
    info = {"postgres": str(version["v"]).split(" on ")[0]}  # type: ignore[index]
    info.update(require_extensions())
    return info


def close_pool() -> None:
    """Close the pool and clear the cache. Called on CLI exit.

    Without this, psycopg's pool workers outlive the interpreter's shutdown
    window and print "couldn't stop thread" warnings over the command's output.
    """
    if pool.cache_info().currsize:
        pool().close()
        pool.cache_clear()


def healthcheck() -> dict[str, str]:
    try:
        return server_info()
    except psycopg.Error as exc:
        raise IndexingError(f"database unreachable at configured DSN: {exc}") from exc
