"""Tests for the frozen question-set loader.

Pinned: gold that points outside the corpus is refused rather than loaded as an
unreachable target (which would read as a retrieval miss), and once a run
exists on the tier the set cannot be edited underneath it.
"""

from __future__ import annotations

import json

import pytest

from agentforge.db.conn import pool
from agentforge.errors import IngestError
from agentforge.ingest import questions

TIER = "pytest_questions"


def write(tmp_path, rows):
    path = tmp_path / "q.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def row(qid="pq1", question="What fuses BM25 and dense?", chunks=("p1:0",)):
    return {
        "query_id": qid, "question": question, "gold_answer": "Hybrid retrieval.",
        "kind": "single", "gold_chunks": list(chunks), "answer_keys": ["hybrid"],
    }


@pytest.fixture
def clean_tier(probe_corpus):
    yield
    with pool().connection() as conn:
        conn.execute("DELETE FROM runs WHERE tier = %s", (TIER,))
        conn.execute("DELETE FROM queries WHERE tier = %s", (TIER,))


def test_loads_span_level_qrels_and_records_the_file_hash(tmp_path, probe_corpus, clean_tier):
    stats = questions.load(probe_corpus, TIER, write(tmp_path, [row(chunks=("p1:0", "p2:0"))]))
    with pool().connection() as conn:
        qrels = conn.execute(
            "SELECT chunk_id, doc_id FROM qrels WHERE query_id = 'pq1' ORDER BY chunk_id"
        ).fetchall()
        meta = conn.execute("SELECT meta FROM queries WHERE query_id = 'pq1'").fetchone()["meta"]
    assert [(r["chunk_id"], r["doc_id"]) for r in qrels] == [("p1:0", None), ("p2:0", None)]
    assert meta["source_sha256"] == stats["sha256"]


def test_gold_chunk_outside_the_corpus_is_refused(tmp_path, probe_corpus, clean_tier):
    with pytest.raises(IngestError, match="not in corpus"):
        questions.load(probe_corpus, TIER, write(tmp_path, [row(chunks=("p1:0", "nope:7"))]))


def test_draft_set_can_be_edited_before_any_run(tmp_path, probe_corpus, clean_tier):
    questions.load(probe_corpus, TIER, write(tmp_path, [row()]))
    questions.load(probe_corpus, TIER, write(tmp_path, [row(question="Edited after review?")]))
    with pool().connection() as conn:
        text = conn.execute("SELECT text FROM queries WHERE query_id = 'pq1'").fetchone()["text"]
    assert text == "Edited after review?"


def test_frozen_set_refuses_edits_once_a_run_exists(tmp_path, probe_corpus, clean_tier):
    path = write(tmp_path, [row()])
    questions.load(probe_corpus, TIER, path)
    with pool().connection() as conn:
        conn.execute(
            "INSERT INTO runs (run_id, config_hash, config_json, git_sha, arm_label, tier) "
            "VALUES ('pytest-run', 'h', '{}', 'sha', 'arm', %s)",
            (TIER,),
        )
    questions.load(probe_corpus, TIER, path)  # identical reload is fine
    with pytest.raises(IngestError, match="frozen"):
        questions.load(probe_corpus, TIER, write(tmp_path, [row(question="Moved goalposts?")]))
