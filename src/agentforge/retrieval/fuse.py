"""Reciprocal Rank Fusion.

Why RRF and not a weighted sum of scores: BM25 scores are unbounded and
corpus-dependent, cosine similarity is bounded in a completely different space.
Adding them requires per-query normalisation (min-max over the returned set),
which is unstable in exactly the case that matters -- when one retriever returns
nothing relevant, its top score normalises to 1.0 and the fusion promotes
garbage. RRF consumes only ranks, so it is scale-free by construction.

    score(d) = sum over retrievers of weight / (k + rank(d))

`k` damps the influence of top ranks; 60 is the value from the original Cormack
et al. formulation and is a tuned arm here rather than a magic number.
"""

from __future__ import annotations

from collections import defaultdict

from agentforge.retrieval.embed import Hit


def rrf(
    ranked_lists: dict[str, list[Hit]],
    weights: dict[str, float],
    k: int = 60,
    limit: int | None = None,
) -> list[Hit]:
    """Fuse named ranked lists. Returns a new ranking, re-ranked from 1."""
    if not ranked_lists:
        raise ValueError("nothing to fuse")
    missing = set(ranked_lists) - set(weights)
    if missing:
        raise ValueError(f"no fusion weight for retriever(s): {sorted(missing)}")

    scores: dict[str, float] = defaultdict(float)
    doc_of: dict[str, str] = {}
    for name, hits in ranked_lists.items():
        weight = weights[name]
        for hit in hits:
            scores[hit.chunk_id] += weight / (k + hit.rank)
            doc_of[hit.chunk_id] = hit.doc_id

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if limit is not None:
        ordered = ordered[:limit]
    return [
        Hit(chunk_id=chunk_id, doc_id=doc_of[chunk_id], score=score, rank=i + 1)
        for i, (chunk_id, score) in enumerate(ordered)
    ]
