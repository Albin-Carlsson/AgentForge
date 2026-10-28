"""Integration tests for the retrieval pipeline.

The pipeline returns EVERY stage's ranking, not just the final one, so that
"what did the reranker actually move?" is answerable per question rather than
only in aggregate. These tests pin that contract, because it is what the
ablation report is built on.
"""

from __future__ import annotations

import pytest

from agentforge.config import RetrievalConfig
from agentforge.errors import IndexingError
from agentforge.retrieval import lexical
from agentforge.retrieval.embed import Embedder
from agentforge.retrieval.pipeline import Retriever

TEST_MODEL = "BAAI/bge-small-en-v1.5"


def cfg(**overrides) -> RetrievalConfig:
    base = {
        "embed_model": TEST_MODEL,
        "candidate_k": 10,
        "final_k": 2,
        "rerank_top_k": 10,
        # Named explicitly: reranking is off by default since the SciFact
        # measurement, so a test of the rerank path must ask for it.
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    }
    return RetrievalConfig(**(base | overrides))


@pytest.fixture
def indexed(probe_corpus) -> str:
    config = cfg()
    lexical.build_index(config)
    Embedder(config.embed_model, config.embed_dim).index_corpus(probe_corpus)
    return probe_corpus


def test_bm25_only_arm_runs_no_dense_stage(indexed):
    config = cfg(retriever="bm25", rerank_model=None)
    trace = Retriever(config, indexed).retrieve("SciFact nDCG@10")
    assert set(trace.stages) == {"bm25"}
    assert trace.final_stage == "bm25"
    assert "dense" not in trace.latency_ms


def test_dense_only_arm_runs_no_lexical_stage(indexed):
    config = cfg(retriever="dense", rerank_model=None)
    trace = Retriever(config, indexed).retrieve("scoring pairs jointly")
    assert set(trace.stages) == {"dense"}
    assert trace.final_stage == "dense"


def test_hybrid_arm_produces_a_fused_stage_from_both(indexed):
    config = cfg(retriever="hybrid", rerank_model=None)
    trace = Retriever(config, indexed).retrieve("reranking improves precision")
    assert {"bm25", "dense", "fused"} <= set(trace.stages)
    assert trace.final_stage == "fused"


def test_reranked_stage_takes_precedence_when_configured(indexed):
    config = cfg(retriever="hybrid")
    trace = Retriever(config, indexed).retrieve("reranking improves precision")
    assert trace.final_stage == "reranked"
    assert "rerank" in trace.latency_ms


def test_every_arm_respects_final_k(indexed):
    for retriever in ("bm25", "dense", "hybrid"):
        for rerank_model in (None, "cross-encoder/ms-marco-MiniLM-L-6-v2"):
            config = cfg(retriever=retriever, rerank_model=rerank_model)
            trace = Retriever(config, indexed).retrieve("retrieval")
            assert len(trace.final) <= config.final_k, (retriever, rerank_model)


def test_ranks_are_contiguous_from_one(indexed):
    trace = Retriever(cfg(retriever="hybrid"), indexed).retrieve("retrieval")
    assert [h.rank for h in trace.final] == list(range(1, len(trace.final) + 1))


def test_trace_records_latency_per_stage(indexed):
    trace = Retriever(cfg(retriever="hybrid"), indexed).retrieve("retrieval")
    assert {"bm25", "dense", "fuse", "rerank"} <= set(trace.latency_ms)
    assert all(v >= 0 for v in trace.latency_ms.values())
    assert trace.total_ms == pytest.approx(sum(trace.latency_ms.values()))


def test_unmatched_lexical_query_yields_an_empty_bm25_arm_not_an_error(indexed):
    """recall=0 is a legitimate result for a query BM25 cannot match."""
    config = cfg(retriever="bm25", rerank_model=None)
    trace = Retriever(config, indexed).retrieve("zzzznonexistentterm")
    assert trace.stages["bm25"] == []
    assert trace.final == []


def test_retrieve_raises_when_the_index_was_never_built(indexed):
    lexical.drop_index()
    with pytest.raises(IndexingError, match="does not exist"):
        Retriever(cfg(retriever="bm25"), indexed).retrieve("retrieval")


def test_reranker_is_skipped_without_candidates(indexed):
    """An empty candidate list must not reach the cross-encoder."""
    config = cfg(retriever="bm25")
    trace = Retriever(config, indexed).retrieve("zzzznonexistentterm")
    assert trace.final == []


def test_concurrent_retrieval_is_safe(indexed):
    """The graph's Researchers retrieve in parallel; unserialised MPS inference aborted."""
    from concurrent.futures import ThreadPoolExecutor

    retriever = Retriever(cfg(retriever="hybrid"), indexed)
    with ThreadPoolExecutor(4) as pool:
        traces = list(pool.map(retriever.retrieve, ["retrieval", "reranking", "dense", "BM25"] * 3))
    assert all(trace.final for trace in traces)
