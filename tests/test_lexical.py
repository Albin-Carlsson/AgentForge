"""Integration tests for BM25 retrieval through pg_search.

Written before the implementation was corrected, against the behaviour actually
observed on pg_search 0.26.3:

  * `key_field` is deprecated and a no-op since 0.26.0.
  * A relation may hold only ONE ParadeDB index, so a tokenizer change is a
    drop-and-recreate, which must therefore be an explicit step.
  * BM25 k1/b are Tantivy constants and are not exposed as index options. The
    config must not claim to sweep them.
  * Stemming is configurable, and does change results.
"""

from __future__ import annotations

import pytest

from agentforge.config import RetrievalConfig
from agentforge.errors import IndexingError
from agentforge.retrieval import lexical


def test_bm25_constants_are_reported_as_untunable():
    """k1/b are documented constants, not swept knobs. Honesty about the engine."""
    params = lexical.bm25_parameters()
    assert params["k1"] == 1.2
    assert params["b"] == 0.75
    assert params["tunable"] == "no"


def test_config_does_not_expose_bm25_k1_or_b():
    """The config must not advertise a sweep pg_search cannot perform."""
    fields = set(RetrievalConfig.model_fields)
    assert "bm25_k1" not in fields
    assert "bm25_b" not in fields


def test_reranking_is_off_by_default_because_it_was_measured_to_lose():
    """On SciFact both rerankers lost to no reranking; the default records that."""
    assert RetrievalConfig().rerank_model is None


def test_config_exposes_the_tokenizer_knobs_that_are_real():
    fields = set(RetrievalConfig.model_fields)
    assert {"bm25_stemmer", "bm25_stopwords"} <= fields


def test_build_index_is_explicit_and_idempotent(probe_corpus):
    # The BM25 index is global to `chunks`, not per corpus, so a sibling test
    # may have left one built. Establish the starting state explicitly.
    lexical.drop_index()
    cfg = RetrievalConfig()
    first = lexical.build_index(cfg)
    assert first.rebuilt is True
    second = lexical.build_index(cfg)
    assert second.rebuilt is False, "rebuilding an unchanged index wastes work silently"
    assert lexical.index_exists()


def test_search_raises_when_index_is_absent(probe_corpus):
    lexical.drop_index()
    with pytest.raises(IndexingError, match="does not exist"):
        lexical.search("retrieval", probe_corpus, RetrievalConfig(), k=5)


def test_search_raises_when_live_index_does_not_match_run_config(probe_corpus):
    """A silent rebuild mid-ablation would thrash between arms. Raise instead."""
    lexical.build_index(RetrievalConfig(bm25_stemmer="English"))
    mismatched = RetrievalConfig(bm25_stemmer=None)
    with pytest.raises(IndexingError, match="does not match"):
        lexical.search("retrieval", probe_corpus, mismatched, k=5)


def test_search_raises_when_corpus_has_no_indexed_rows():
    """An unindexed corpus is a defect; it must not read as 'nothing relevant'."""
    lexical.build_index(RetrievalConfig())
    with pytest.raises(IndexingError, match="no indexed rows"):
        lexical.search("retrieval", "corpus_that_does_not_exist", RetrievalConfig(), k=5)


def test_no_match_returns_empty_and_does_not_raise(probe_corpus):
    """'Retriever found nothing' is a measurement (recall=0), not an error.

    Raising here would drop exactly the hard queries from the evaluation and
    silently inflate every reported recall number.
    """
    cfg = RetrievalConfig()
    lexical.build_index(cfg)
    hits = lexical.search("zzzznonexistentterm", probe_corpus, cfg, k=5)
    assert hits == []


def test_search_ranks_exact_identifier_matches_first(probe_corpus):
    cfg = RetrievalConfig()
    lexical.build_index(cfg)
    hits = lexical.search("SciFact nDCG@10", probe_corpus, cfg, k=5)
    assert hits, "BM25 should match the rare identifiers it is good at"
    assert hits[0].chunk_id == "p1:0"
    assert hits[0].rank == 1
    assert hits[0].score > 0


def test_search_is_scoped_to_the_requested_corpus(probe_corpus):
    cfg = RetrievalConfig()
    lexical.build_index(cfg)
    hits = lexical.search("retrieval", probe_corpus, cfg, k=10)
    assert all(h.chunk_id.startswith(("p1", "p2", "p3")) for h in hits)


def test_stemming_is_a_real_arm(probe_corpus):
    """With the English stemmer, 'evaluating' must match 'evaluated'."""
    stemmed = RetrievalConfig(bm25_stemmer="English")
    lexical.build_index(stemmed)
    assert lexical.search("evaluating", probe_corpus, stemmed, k=5)

    unstemmed = RetrievalConfig(bm25_stemmer=None)
    lexical.build_index(unstemmed)
    assert lexical.search("evaluating", probe_corpus, unstemmed, k=5) == []


def test_punctuation_heavy_query_does_not_break_the_parser(probe_corpus):
    """Raw question text must not reach the Tantivy query parser."""
    cfg = RetrievalConfig()
    lexical.build_index(cfg)
    hits = lexical.search("What is nDCG@10, and why BM25?", probe_corpus, cfg, k=5)
    assert hits
