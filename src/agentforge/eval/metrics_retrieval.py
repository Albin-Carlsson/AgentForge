"""Retrieval metrics, computed from ranks against gold judgements.

No LLM is involved anywhere in this module, which is the single most important
economic fact about this project: the entire retrieval ablation -- chunk size,
embedding model, fusion weights, reranker on/off, BM25 vs dense vs hybrid -- is
free to run and free to re-run. That is what makes "every parameter was tuned by
measurement" affordable rather than aspirational.

BEIR qrels are document-level, the arXiv tier's gold labels are span-level, so
every metric takes the granularity it is scored at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalScores:
    recall_at_k: float
    ndcg_at_k: float
    mrr: float
    k: int
    n_relevant: int

    def as_row(self) -> dict[str, float | int]:
        return {
            f"recall@{self.k}": round(self.recall_at_k, 4),
            f"nDCG@{self.k}": round(self.ndcg_at_k, 4),
            "MRR": round(self.mrr, 4),
            "n_relevant": self.n_relevant,
        }


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items appearing in the top k.

    Undefined with no relevant items -- that is a broken qrel, not a score of
    zero, so it raises rather than silently contributing 0.0 to a mean.
    """
    if not relevant:
        raise ValueError("recall is undefined for a query with no relevant documents")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: list[str], gains: dict[str, int], k: int) -> float:
    """Binary or graded nDCG with the standard 2^gain - 1 formulation."""
    if not gains:
        raise ValueError("nDCG is undefined for a query with no graded judgements")

    dcg = sum(
        (2 ** gains.get(item, 0) - 1) / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
    )
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(
        (2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(ideal, start=1)
    )
    return dcg / idcg if idcg > 0 else 0.0


def score_query(
    retrieved: list[str], gains: dict[str, int], k: int
) -> RetrievalScores:
    relevant = {item for item, gain in gains.items() if gain > 0}
    return RetrievalScores(
        recall_at_k=recall_at_k(retrieved, relevant, k),
        ndcg_at_k=ndcg_at_k(retrieved, gains, k),
        mrr=reciprocal_rank(retrieved, relevant),
        k=k,
        n_relevant=len(relevant),
    )
