import math

import pytest

from agentforge.eval.metrics_retrieval import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_query,
)


def test_recall_counts_only_top_k():
    assert recall_at_k(["a", "b", "c", "d"], {"a", "d"}, k=2) == 0.5
    assert recall_at_k(["a", "b", "c", "d"], {"a", "d"}, k=4) == 1.0


def test_recall_with_no_relevant_docs_raises_rather_than_scoring_zero():
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k(["a"], set(), k=1)


def test_reciprocal_rank_uses_first_hit():
    assert reciprocal_rank(["x", "y", "a"], {"a", "y"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"z"}) == 0.0


def test_ndcg_is_one_for_ideal_ordering():
    gains = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], gains, k=2) == pytest.approx(1.0)


def test_ndcg_penalises_inverted_ordering():
    gains = {"a": 2, "b": 1}
    inverted = ndcg_at_k(["b", "a"], gains, k=2)
    assert 0.0 < inverted < 1.0


def test_ndcg_matches_hand_computed_value():
    # single relevant doc with gain 1 at rank 2:
    # DCG = (2^1-1)/log2(3) = 0.6309; IDCG = 1/log2(2) = 1.0
    gains = {"a": 1}
    assert ndcg_at_k(["x", "a"], gains, k=10) == pytest.approx(1 / math.log2(3), abs=1e-6)


def test_score_query_bundles_all_three():
    scores = score_query(["a", "x", "b"], {"a": 1, "b": 1}, k=3)
    assert scores.recall_at_k == 1.0
    assert scores.mrr == 1.0
    assert scores.n_relevant == 2
    assert "recall@3" in scores.as_row()
