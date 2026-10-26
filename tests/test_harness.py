"""Tests for the experiment harness.

What is being pinned here is the attribution discipline, not the plumbing:

  * a run is identified by its config hash, and the config is stored with it;
  * results are stored per question, never pre-aggregated;
  * two runs are comparable only when exactly one knob differs;
  * scoring granularity is read from the qrels rather than assumed, because
    BEIR judges documents and the arXiv tier judges spans.
"""

from __future__ import annotations

import pytest

from agentforge.config import IngestConfig, RetrievalConfig, RunConfig, assert_comparable
from agentforge.errors import IncomparableRuns, MeasurementError
from agentforge.eval import harness
from agentforge.retrieval import lexical
from agentforge.retrieval.embed import Embedder

TEST_MODEL = "BAAI/bge-small-en-v1.5"


def make_config(**retrieval_overrides) -> RunConfig:
    base = {
        "embed_model": TEST_MODEL,
        "candidate_k": 10,
        "final_k": 3,
        "rerank_top_k": 10,
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }
    return RunConfig(
        ingest=IngestConfig(corpus_id="pytest_probe"),
        retrieval=RetrievalConfig(**(base | retrieval_overrides)),
    )


@pytest.fixture
def indexed(probe_tier, probe_corpus) -> str:
    config = make_config()
    lexical.build_index(config.retrieval)
    Embedder(config.retrieval.embed_model, config.retrieval.embed_dim).index_corpus(
        probe_corpus
    )
    return probe_tier


def test_run_records_its_config_hash_and_git_sha(indexed):
    config = make_config(retriever="bm25", rerank_model=None)
    run = harness.execute_retrieval_run(config, tier=indexed, arm_label="bm25-only")
    stored = harness.load_run(run.run_id)
    assert stored["config_hash"] == config.config_hash
    assert stored["arm_label"] == "bm25-only"
    assert stored["git_sha"]


def test_results_are_stored_per_question_for_every_stage(indexed):
    config = make_config(retriever="hybrid")
    run = harness.execute_retrieval_run(config, tier=indexed, arm_label="hybrid+rerank")
    rows = harness.load_retrieval_rows(run.run_id)
    assert {r["stage"] for r in rows} >= {"bm25", "dense", "fused", "reranked"}
    assert {r["query_id"] for r in rows} == {"probe:q1", "probe:q2", "probe:q3"}


def test_scores_are_computed_per_question_not_only_in_aggregate(indexed):
    config = make_config(retriever="hybrid")
    run = harness.execute_retrieval_run(config, tier=indexed, arm_label="hybrid+rerank")
    per_query = harness.per_query_metric(run.run_id, metric="recall", k=3)
    assert set(per_query) == {"probe:q1", "probe:q2", "probe:q3"}


def test_hybrid_retrieves_every_relevant_document_on_the_probe_tier(indexed):
    """A sanity floor: three queries, three distinct targets, all findable."""
    config = make_config(retriever="hybrid")
    run = harness.execute_retrieval_run(config, tier=indexed, arm_label="hybrid+rerank")
    scores = harness.score_run(run.run_id, k=3)
    assert scores["recall@3"] == pytest.approx(1.0)
    assert scores["n_queries"] == 3


def test_granularity_is_read_from_the_qrels(indexed):
    """BEIR judges documents; the arXiv tier will judge spans. Never assumed."""
    assert harness.scoring_granularity(indexed) == "doc_id"


def test_latency_from_a_cached_run_is_refused(indexed):
    """Cached calls return in microseconds; reporting that as p95 is fiction."""
    config = make_config()
    run = harness.execute_retrieval_run(
        config, tier=indexed, arm_label="hybrid+rerank", cache_bypass=False
    )
    with pytest.raises(MeasurementError, match="cache"):
        harness.latency_percentile(run.run_id, q=95)


def test_latency_is_available_from_a_cache_bypassed_run(indexed):
    config = make_config()
    run = harness.execute_retrieval_run(
        config, tier=indexed, arm_label="hybrid+rerank", cache_bypass=True
    )
    assert harness.latency_percentile(run.run_id, q=95) > 0


def test_single_knob_ladder_steps_are_comparable():
    bm25 = make_config(retriever="bm25", rerank_model=None)
    dense = make_config(retriever="dense", rerank_model=None)
    hybrid = make_config(retriever="hybrid", rerank_model=None)
    reranked = make_config(retriever="hybrid")

    assert assert_comparable(bm25, dense) == "retrieval.retriever"
    assert assert_comparable(dense, hybrid) == "retrieval.retriever"
    assert assert_comparable(hybrid, reranked) == "retrieval.rerank_model"


def test_two_knob_leap_is_refused():
    """dense-only vs hybrid+rerank moves two knobs: the delta is unattributable."""
    dense = make_config(retriever="dense", rerank_model=None)
    reranked = make_config(retriever="hybrid")
    with pytest.raises(IncomparableRuns, match="2 fields"):
        assert_comparable(dense, reranked)


def test_comparing_a_config_against_itself_is_refused():
    config = make_config()
    with pytest.raises(IncomparableRuns, match="identical"):
        assert_comparable(config, config)


def test_compare_runs_reports_a_paired_delta(indexed):
    no_rerank = make_config(retriever="hybrid", rerank_model=None)
    with_rerank = make_config(retriever="hybrid")
    a = harness.execute_retrieval_run(no_rerank, tier=indexed, arm_label="hybrid")
    b = harness.execute_retrieval_run(with_rerank, tier=indexed, arm_label="hybrid+rerank")

    comparison = harness.compare_runs(a.run_id, b.run_id, metric="ndcg", k=3)
    assert comparison.knob == "retrieval.rerank_model"
    assert comparison.delta.n == 3


def test_scoring_beyond_final_k_is_refused(indexed):
    """recall@10 is meaningless when the pipeline returns at most 8 chunks.

    This guard exists because the first real ablation run reported recall@10 for
    arms configured with final_k=8, which structurally capped every arm at
    recall@8 and made the reranker look worse than it was.
    """
    config = make_config(retriever="hybrid", final_k=3)
    run = harness.execute_retrieval_run(config, tier=indexed, arm_label="hybrid+rerank")
    with pytest.raises(MeasurementError, match="final_k"):
        harness.score_run(run.run_id, k=10)
    # At or below final_k it scores normally.
    assert harness.score_run(run.run_id, k=3)["n_queries"] == 3
