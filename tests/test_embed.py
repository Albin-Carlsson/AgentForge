"""Integration tests for dense retrieval through pgvector.

Uses bge-small (133 MB) rather than the shipped bge-base (440 MB) to keep the
suite fast; the dimension is read from the config either way, which is the
behaviour under test.

The asymmetry of the empty case matters and is why these tests differ in shape
from the lexical ones: BM25 can legitimately match nothing, but a nearest-
neighbour search over a populated table always returns k rows. So for dense
retrieval, "empty" can only mean the corpus was never embedded -- a defect.
"""

from __future__ import annotations

import pytest

from agentforge.config import RetrievalConfig
from agentforge.errors import IndexingError
from agentforge.retrieval.embed import Embedder, table_for

TEST_MODEL = "BAAI/bge-small-en-v1.5"


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    cfg = RetrievalConfig(embed_model=TEST_MODEL)
    return Embedder(cfg.embed_model, cfg.embed_dim)


def test_table_name_is_derived_from_the_model():
    assert table_for("BAAI/bge-small-en-v1.5") == "emb_baai_bge_small_en_v1_5"
    assert table_for("BAAI/bge-base-en-v1.5") == "emb_baai_bge_base_en_v1_5"


def test_each_model_gets_its_own_table():
    """A pgvector column is fixed-dimension and dimension is an arm."""
    assert table_for("BAAI/bge-small-en-v1.5") != table_for("BAAI/bge-m3")


def test_unknown_model_raises_rather_than_guessing_a_dimension():
    from agentforge.errors import ConfigError

    with pytest.raises(ConfigError, match="unknown embed_model"):
        _ = RetrievalConfig(embed_model="some/unregistered-model").embed_dim


def test_model_without_a_query_prefix_policy_raises():
    """bge models are asymmetric; guessing would silently cap dense recall."""
    with pytest.raises(IndexingError, match="query-prefix policy"):
        Embedder("fake/unregistered-model", 384)


def test_search_raises_when_the_corpus_was_never_embedded(embedder, probe_corpus):
    embedder.ensure_table()
    with pytest.raises(IndexingError, match="not embedded"):
        embedder.search("retrieval", "corpus_that_does_not_exist", k=5)


def test_index_corpus_embeds_every_chunk(embedder, probe_corpus):
    stats = embedder.index_corpus(probe_corpus)
    assert stats["chunks_encoded"] == 3.0
    assert stats["encode_seconds"] > 0

    # Idempotent: a second pass has nothing left to encode.
    again = embedder.index_corpus(probe_corpus)
    assert again["chunks_encoded"] == 0.0


def test_dense_search_finds_a_paraphrase_with_no_shared_terms(embedder, probe_corpus):
    """This is the case that justifies dense retrieval existing at all.

    The query shares no content word with the target chunk, so BM25 would score
    it at zero.
    """
    embedder.index_corpus(probe_corpus)
    hits = embedder.search("scoring pairs together rather than separately", probe_corpus, k=3)
    assert hits
    assert hits[0].chunk_id == "p2:0"  # the cross-encoder chunk
    assert hits[0].rank == 1
    assert 0.0 <= hits[0].score <= 1.0


def test_query_prefix_is_applied_for_bge(embedder):
    assert embedder.query_prefix.startswith("Represent this sentence")


def test_declared_dimension_mismatch_raises(embedder, monkeypatch):
    """A model silently producing a different width must not be written."""
    monkeypatch.setattr(embedder, "dim", 999)
    with pytest.raises(IndexingError, match="declares"):
        embedder.encode_passages(["some text"])
