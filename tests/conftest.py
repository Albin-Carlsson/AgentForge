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

import pytest

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


@pytest.fixture
def probe_corpus() -> str:
    """A tiny corpus with known contents, torn down after each test."""
    import json

    documents = [
        ("p1", "Hybrid retrieval", "Hybrid retrieval combines BM25 and dense scores, "
                                   "evaluated on BEIR SciFact with nDCG@10."),
        ("p2", "Cross-encoder reranking", "A cross-encoder rescores candidates jointly "
                                          "with the query and improves precision."),
        ("p3", "Dense bi-encoders", "Bi-encoders embed queries and passages separately, "
                                    "which is fast but weaker on rare identifiers."),
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
