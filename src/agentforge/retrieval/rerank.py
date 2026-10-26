"""Cross-encoder reranking.

A bi-encoder embeds query and passage independently, so it never sees them
together; a cross-encoder scores the pair jointly and is substantially more
accurate at the top of the list. The cost is that it cannot be precomputed --
every candidate needs a forward pass at query time, which is why it reranks a
shortlist rather than the corpus.

MiniLM-L-6 is 90 MB and scores 50 candidates in tens of milliseconds on an M3.
The small size is the point: it makes the latency claim concrete and the
quality/latency curve cheap to measure (`bge-reranker-base` is the second arm).
"""

from __future__ import annotations

import time
from functools import lru_cache

from sentence_transformers import CrossEncoder

from agentforge.db.conn import pool
from agentforge.errors import RetrievalError
from agentforge.retrieval.embed import Hit, device
from agentforge.settings import settings


@lru_cache(maxsize=2)
def _cross_encoder(name: str) -> CrossEncoder:
    return CrossEncoder(name, device=device(), cache_folder=str(settings().model_dir))


def _chunk_texts(chunk_ids: list[str]) -> dict[str, str]:
    with pool().connection() as conn:
        rows = conn.execute(
            "SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)", (chunk_ids,)
        ).fetchall()
    texts = {r["chunk_id"]: r["text"] for r in rows}
    absent = [c for c in chunk_ids if c not in texts]
    if absent:
        # A retrieved chunk_id that is not in the corpus means the index and the
        # table have diverged. Raising beats reranking a shorter list and
        # reporting the result as if it were complete.
        raise RetrievalError(f"retrieved chunk_ids absent from chunks table: {absent[:5]}")
    return texts


def rerank(
    query: str, hits: list[Hit], model: str, top_k: int, final_k: int
) -> tuple[list[Hit], float]:
    """Rescore the top `top_k` candidates jointly with the query.

    Returns the reranked top `final_k` and the elapsed milliseconds, which is a
    reported column -- the whole claim is that the cross-encoder buys quality for
    a known latency cost.
    """
    if not hits:
        raise RetrievalError("rerank called with no candidates")

    shortlist = hits[:top_k]
    texts = _chunk_texts([h.chunk_id for h in shortlist])

    started = time.perf_counter()
    scores = _cross_encoder(model).predict(
        [(query, texts[h.chunk_id]) for h in shortlist], show_progress_bar=False
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    ordered = sorted(
        zip(shortlist, (float(s) for s in scores), strict=True),
        key=lambda pair: (-pair[1], pair[0].chunk_id),
    )[:final_k]

    return [
        Hit(chunk_id=hit.chunk_id, doc_id=hit.doc_id, score=score, rank=i + 1)
        for i, (hit, score) in enumerate(ordered)
    ], elapsed_ms
