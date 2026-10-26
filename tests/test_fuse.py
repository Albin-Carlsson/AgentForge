from agentforge.retrieval.embed import Hit
from agentforge.retrieval.fuse import rrf


def hits(*chunk_ids: str) -> list[Hit]:
    return [
        Hit(chunk_id=c, doc_id=c.split(":")[0], score=1.0 / (i + 1), rank=i + 1)
        for i, c in enumerate(chunk_ids)
    ]


def test_rrf_promotes_documents_both_retrievers_agree_on():
    fused = rrf(
        {"bm25": hits("a:1", "b:1", "c:1"), "dense": hits("c:1", "a:1", "d:1")},
        weights={"bm25": 1.0, "dense": 1.0},
    )
    # a is ranked 1 and 2; c is 3 and 1. Both beat b and d, which appear once.
    assert {h.chunk_id for h in fused[:2]} == {"a:1", "c:1"}
    assert fused[0].rank == 1 and fused[1].rank == 2


def test_rrf_is_scale_free():
    """A retriever with wildly larger raw scores must not dominate."""
    huge = [Hit(chunk_id="x:1", doc_id="x", score=98_000.0, rank=1)]
    small = [Hit(chunk_id="y:1", doc_id="y", score=0.0001, rank=1)]
    fused = rrf({"bm25": huge, "dense": small}, weights={"bm25": 1.0, "dense": 1.0})
    assert fused[0].score == fused[1].score  # equal ranks -> equal contribution


def test_weights_shift_the_balance():
    fused = rrf(
        {"bm25": hits("a:1"), "dense": hits("b:1")},
        weights={"bm25": 3.0, "dense": 1.0},
    )
    assert fused[0].chunk_id == "a:1"


def test_limit_truncates_and_reranks():
    fused = rrf({"bm25": hits("a:1", "b:1", "c:1")}, weights={"bm25": 1.0}, limit=2)
    assert [h.rank for h in fused] == [1, 2]


def test_missing_weight_raises():
    import pytest

    with pytest.raises(ValueError, match="no fusion weight"):
        rrf({"bm25": hits("a:1")}, weights={})
