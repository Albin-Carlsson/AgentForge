"""The retrieval pipeline: one entry point, used by the harness and the agents alike.

Every stage's ranking is returned, not just the final one. That is deliberate:
storing `bm25`, `dense`, `fused` and `reranked` separately makes "what did the
reranker actually move?" answerable per question rather than only in aggregate,
which is where the useful diagnostics live.

Note what this module does NOT do: it never builds an index. Index construction
is an explicit step the harness performs between arms, because a pipeline that
rebuilt an index on demand would thrash during an ablation and bill work nobody
asked for. Querying against a stale or absent index raises.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agentforge.config import RetrievalConfig
from agentforge.errors import RetrievalError
from agentforge.retrieval import lexical
from agentforge.retrieval.embed import Embedder, Hit
from agentforge.retrieval.fuse import rrf
from agentforge.retrieval.rerank import rerank

# Most specific stage first: the final ranking is the last stage that ran.
_STAGE_PRECEDENCE = ("reranked", "fused", "dense", "bm25")


@dataclass(frozen=True)
class RetrievalTrace:
    """Per-stage rankings and timings for one query."""

    stages: dict[str, list[Hit]] = field(default_factory=dict)
    latency_ms: dict[str, float] = field(default_factory=dict)

    @property
    def final_stage(self) -> str:
        for name in _STAGE_PRECEDENCE:
            if name in self.stages:
                return name
        raise RetrievalError("trace contains no stages")

    @property
    def final(self) -> list[Hit]:
        return self.stages[self.final_stage]

    @property
    def total_ms(self) -> float:
        return sum(self.latency_ms.values())


class Retriever:
    def __init__(self, config: RetrievalConfig, corpus_id: str) -> None:
        self.config = config
        self.corpus_id = corpus_id
        self.embedder = (
            Embedder(config.embed_model, config.embed_dim)
            if config.retriever in ("dense", "hybrid")
            else None
        )

    def retrieve(self, query: str) -> RetrievalTrace:
        cfg = self.config
        stages: dict[str, list[Hit]] = {}
        timings: dict[str, float] = {}

        if cfg.retriever in ("bm25", "hybrid"):
            started = time.perf_counter()
            stages["bm25"] = lexical.search(query, self.corpus_id, cfg, cfg.candidate_k)
            timings["bm25"] = (time.perf_counter() - started) * 1000

        if cfg.retriever in ("dense", "hybrid"):
            if self.embedder is None:  # pragma: no cover - constructed above
                raise RetrievalError(f"no embedder for retriever {cfg.retriever!r}")
            started = time.perf_counter()
            stages["dense"] = self.embedder.search(query, self.corpus_id, cfg.candidate_k)
            timings["dense"] = (time.perf_counter() - started) * 1000

        if cfg.retriever == "hybrid":
            started = time.perf_counter()
            stages["fused"] = rrf(
                {"bm25": stages["bm25"], "dense": stages["dense"]},
                weights={"bm25": cfg.bm25_weight, "dense": cfg.dense_weight},
                k=cfg.rrf_k,
            )
            timings["fuse"] = (time.perf_counter() - started) * 1000

        candidate_stage = "fused" if cfg.retriever == "hybrid" else cfg.retriever
        candidates = stages[candidate_stage]

        if cfg.rerank_model is None:
            # "No reranker" is a declared arm, not a degraded path: truncating
            # the candidate list to final_k is precisely what that arm means.
            stages[candidate_stage] = candidates[: cfg.final_k]
        elif not candidates:
            # Nothing matched. The cross-encoder has nothing to score, and an
            # empty reranked stage is the honest record of that -- recall 0,
            # not an error.
            stages["reranked"] = []
            timings["rerank"] = 0.0
        else:
            reranked, elapsed = rerank(
                query,
                candidates,
                model=cfg.rerank_model,
                top_k=cfg.rerank_top_k,
                final_k=cfg.final_k,
            )
            stages["reranked"] = reranked
            timings["rerank"] = elapsed

        return RetrievalTrace(stages=stages, latency_ms=timings)
