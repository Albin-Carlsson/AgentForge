"""Test fixtures.

These are integration tests against a real ParadeDB instance, deliberately. The
entire point of the lexical module is that `pg_search` behaves the way the code
believes it does -- a mocked BM25 index would prove nothing about the claim on
the resume, and the real syntax has already diverged from what was written
against documentation once in this project.

If the database is not running these tests FAIL rather than skip. A skipped test
is a silent pass, and CI is specified to run the retrieval tier.
"""

from __future__ import annotations

import os

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from agentforge.settings import settings


def _use_test_database() -> None:
    """Point the whole suite at `<db>_test`, never at the real database.

    The fixtures below delete per-test state -- llm_cache among it. Run against
    the real database, that wiped every cached model call (and the judge's
    reference pairs) on each `make test`, which silently turns a resumable
    multi-day run into a re-billed one. Derived, not configured, so there is no
    way to forget it; and asserted, so a DSN already naming a test database
    cannot be doubled into `_test_test` or, worse, left pointing at the real one.
    """
    real = conninfo_to_dict(settings().pg_dsn)
    name = real["dbname"]
    if name.endswith("_test"):
        raise RuntimeError(f"AF_PG_DSN already names a test database ({name}); set the real one")
    test_dsn = make_conninfo(**{**real, "dbname": f"{name}_test"})
    admin = make_conninfo(**{**real, "dbname": "postgres"})
    with psycopg.connect(admin, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (f"{name}_test",)
        ).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{name}_test"')
    os.environ["AF_PG_DSN"] = test_dsn
    settings.cache_clear()
    assert conninfo_to_dict(settings().pg_dsn)["dbname"] == f"{name}_test"


_use_test_database()

from agentforge.db.conn import apply_schema, healthcheck, pool
from agentforge.errors import AgentForgeError

PROBE_CORPUS = "pytest_probe"
PROBE_TIER = "pytest_tier"


@pytest.fixture(scope="session", autouse=True)
def database() -> dict[str, str]:
    try:
        info = healthcheck()
    except AgentForgeError as exc:
        raise RuntimeError(
            f"the test database is not reachable: {exc}\n"
            "Start it with:  docker-compose up -d db\n"
            "These tests do not skip: a skipped test is a silent pass."
        ) from exc
    apply_schema()
    return info


# Tables holding per-test state. Reference pairs in particular leak loudly:
# a leftover pair makes a scripted judge run out of verdicts.
_EPHEMERAL_TABLES = ("llm_cache", "judge_reference", "judge_validation", "research_jobs")


@pytest.fixture
def db(database) -> None:
    """Truncates per-test state so each test starts from a known database."""
    with pool().connection() as conn:
        for table in _EPHEMERAL_TABLES:
            conn.execute(f"DELETE FROM {table}")


@pytest.fixture
def probe_corpus() -> str:
    """A tiny corpus with known contents, torn down after each test."""
    import json

    documents = [
        (
            "p1",
            "Hybrid retrieval",
            (
                "Hybrid retrieval combines BM25 and dense scores, "
                "evaluated on BEIR SciFact with nDCG@10."
            ),
        ),
        (
            "p2",
            "Cross-encoder reranking",
            (
                "A cross-encoder rescores candidates jointly with the query "
                "and improves precision."
            ),
        ),
        (
            "p3",
            "Dense bi-encoders",
            (
                "Bi-encoders embed queries and passages separately, "
                "which is fast but weaker on rare identifiers."
            ),
        ),
    ]
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO corpora (corpus_id, kind, ingest_config) VALUES (%s,'beir',%s) "
            "ON CONFLICT (corpus_id) DO NOTHING",
            (PROBE_CORPUS, json.dumps({"probe": True})),
        )
        for doc_id, title, text in documents:
            cur.execute(
                "INSERT INTO documents (doc_id, corpus_id, source_type, source_ref, title) "
                "VALUES (%s,%s,'beir',%s,%s) ON CONFLICT (doc_id) DO NOTHING",
                (doc_id, PROBE_CORPUS, f"probe#{doc_id}", title),
            )
            cur.execute(
                "INSERT INTO chunks (chunk_id, doc_id, corpus_id, ordinal, text, embed_text) "
                "VALUES (%s,%s,%s,0,%s,%s) ON CONFLICT (chunk_id) DO NOTHING",
                (f"{doc_id}:0", doc_id, PROBE_CORPUS, text, f"{title}. {text}"),
            )
    yield PROBE_CORPUS
    with pool().connection() as conn:
        conn.execute("DELETE FROM corpora WHERE corpus_id = %s", (PROBE_CORPUS,))


@pytest.fixture
def probe_tier(probe_corpus: str) -> str:
    """Queries and document-level qrels over the probe corpus.

    Document-level, like BEIR's, so the harness's granularity detection is
    exercised by the tests rather than only by the real tier.
    """
    queries = [
        ("probe:q1", "SciFact nDCG@10 hybrid retrieval", {"p1": 1}),
        ("probe:q2", "scoring pairs jointly with the query", {"p2": 1}),
        ("probe:q3", "embedding queries and passages separately", {"p3": 1}),
    ]
    with pool().connection() as conn, conn.cursor() as cur:
        for query_id, text, qrels in queries:
            cur.execute(
                "INSERT INTO queries (query_id, tier, corpus_id, text) "
                "VALUES (%s,%s,%s,%s) ON CONFLICT (query_id) DO NOTHING",
                (query_id, PROBE_TIER, probe_corpus, text),
            )
            for doc_id, relevance in qrels.items():
                cur.execute(
                    "INSERT INTO qrels (query_id, doc_id, relevance) VALUES (%s,%s,%s)",
                    (query_id, doc_id, relevance),
                )
    yield PROBE_TIER
    with pool().connection() as conn:
        conn.execute("DELETE FROM queries WHERE tier = %s", (PROBE_TIER,))
