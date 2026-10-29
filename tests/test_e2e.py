"""Tests for the end-to-end runner and its deterministic correctness score.

Pinned: the control arm stores an answer with no citations; correctness is
key-fact recall with normalisation but no leniency beyond it; a partial run is
refused by scoring instead of being averaged over the questions it reached; and
cost is priced at list for cached calls too, so a replay reports the same cost.
"""

from __future__ import annotations

import json

import pytest

from agentforge.config import MODEL_PRICES, GenerationConfig, IngestConfig, RunConfig
from agentforge.db.conn import pool
from agentforge.errors import MeasurementError, SchemaViolation
from agentforge.eval import e2e
from agentforge.eval.answer_keys import key_found, key_recall

TIER = "pytest_e2e"


def test_key_matching_normalises_separators_and_percent_spacing():
    assert key_found("It has 1860 questions and 33 % lower SE", "1,860")
    assert key_found("33 % lower", "33%")
    assert key_found("a V-shaped curve", "v-shaped|v shape")
    assert not key_found("seventy-six percent", "76%")
    assert key_recall("0.774 vs 0.678", ["0.774", "0.678", "0.758"]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        key_recall("anything", [])


@pytest.fixture
def tier(db, probe_corpus):
    with pool().connection() as conn:
        for qid, text, keys in [("pe1", "Recall@10 of BM25?", ["0.678"]),
                                ("pe2", "How many questions?", ["1,860", "581"])]:
            conn.execute(
                "INSERT INTO queries (query_id, tier, corpus_id, text, meta) VALUES (%s,%s,%s,%s,%s)",
                (qid, TIER, probe_corpus, text, json.dumps({"answer_keys": keys})),
            )
    yield TIER
    with pool().connection() as conn:
        conn.execute("DELETE FROM runs WHERE tier = %s", (TIER,))
        conn.execute("DELETE FROM queries WHERE tier = %s", (TIER,))


class Scripted:
    def __init__(self, answers):
        self.answers = list(answers)

    def __call__(self, *, model, system, messages, max_tokens, schema, thinking):
        if not self.answers:
            raise SchemaViolation("scripted API failure part-way through the run")
        return {"parsed": {"answer": self.answers.pop(0)}, "input_tokens": 100, "output_tokens": 50}


def control_config():
    return RunConfig(ingest=IngestConfig(corpus_id="pytest_probe"),
                     generation=GenerationConfig(graph="none"))


def test_control_arm_stores_answers_and_scores_correctness(tier):
    handle = e2e.execute_e2e_run(control_config(), tier, "control",
                                 Scripted(["BM25 reaches 0.678.", "1860 questions on 581 items."]))
    scores = e2e.score_e2e_run(handle.run_id)
    assert scores.correctness == 1.0
    assert scores.citations == 0 and scores.groundedness is None
    per_in, per_out = MODEL_PRICES[control_config().generation.model]
    assert scores.list_cost_usd == pytest.approx(2 * (100 * per_in + 50 * per_out) / 1e6)


def test_a_replayed_run_reports_the_same_list_cost(tier):
    config = control_config()
    first = e2e.execute_e2e_run(config, tier, "control", Scripted(["0.678", "1,860 and 581"]))
    # Empty script: every call must come from the cache, or the transport raises.
    replay = e2e.execute_e2e_run(config, tier, "control", Scripted([]))
    assert e2e.score_e2e_run(replay.run_id).list_cost_usd == pytest.approx(
        e2e.score_e2e_run(first.run_id).list_cost_usd
    )


def test_a_run_stopped_part_way_is_unscorable(tier):
    with pytest.raises(SchemaViolation):
        e2e.execute_e2e_run(control_config(), tier, "control", Scripted(["only one answer"]))
    with pool().connection() as conn:
        run_id = conn.execute("SELECT run_id FROM runs WHERE tier = %s", (tier,)).fetchone()["run_id"]
    with pytest.raises(MeasurementError, match="unfinished"):
        e2e.score_e2e_run(run_id)
