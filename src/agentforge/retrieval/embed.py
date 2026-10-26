"""Dense retrieval: local embeddings in pgvector.

Why local rather than an API: embedding model and chunk size are experiment
arms, and changing either requires re-embedding the whole corpus. That has to be
a CPU-minutes decision rather than an API-bill decision, or the ablation simply
does not get run. Anthropic serves no embedding model, so this was a choice
either way.

Embeddings live in their own table per model, not in `chunks`. A pgvector column
is fixed-dimension and the dimension varies by arm (384/768/1024), so each model
gets `emb_<slug>` created on demand.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from agentforge.db.conn import pool
from agentforge.errors import IndexingError
from agentforge.settings import settings

# bge-*-en-v1.5 is trained with an asymmetric instruction: queries get a prefix,
# passages do not. Omitting it measurably costs recall, and it is the kind of
# model-specific detail that silently caps a dense baseline.
_QUERY_PREFIX = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-m3": "",  # m3 is symmetric; no instruction
}


def device() -> str:
    """Recorded on every run: it is part of what a latency number means."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def table_for(embed_model: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", embed_model.lower()).strip("_")
    return f"emb_{slug}"


@lru_cache(maxsize=4)
def _model(name: str) -> SentenceTransformer:
    return SentenceTransformer(
        name, device=device(), cache_folder=str(settings().model_dir)
    )


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    score: float
    rank: int


class Embedder:
    def __init__(self, embed_model: str, dim: int) -> None:
        self.embed_model = embed_model
        self.dim = dim
        self.table = table_for(embed_model)
        self.query_prefix = _QUERY_PREFIX.get(embed_model)
        if self.query_prefix is None:
            raise IndexingError(
                f"no query-prefix policy recorded for {embed_model!r}. bge models are "
                "asymmetric; guessing would silently cap dense recall."
            )

    # -- indexing ---------------------------------------------------------

    def ensure_table(self) -> None:
        with pool().connection() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    chunk_id  text PRIMARY KEY
                              REFERENCES chunks(chunk_id) ON DELETE CASCADE,
                    corpus_id text NOT NULL,
                    embedding vector({self.dim}) NOT NULL
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_corpus_idx "
                f"ON {self.table} (corpus_id)"
            )

    def build_index(self) -> float:
        """HNSW over cosine distance. Returns build seconds, which is a reported column."""
        started = time.perf_counter()
        with pool().connection() as conn:
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw ON {self.table} "
                f"USING hnsw (embedding vector_cosine_ops)"
            )
        return time.perf_counter() - started

    def encode_passages(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        vectors = _model(self.embed_model).encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # cosine on unit vectors
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        if vectors.shape[1] != self.dim:
            raise IndexingError(
                f"{self.embed_model} produced dim {vectors.shape[1]}, "
                f"config declares {self.dim}"
            )
        return vectors.tolist()

    def encode_query(self, text: str) -> list[float]:
        vector = _model(self.embed_model).encode(
            self.query_prefix + text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vector.tolist()

    def index_corpus(self, corpus_id: str, batch_size: int = 256) -> dict[str, float]:
        """Embed every chunk of a corpus. Fails loudly; never partially indexes."""
        self.ensure_table()
        with pool().connection() as conn:
            pending = conn.execute(
                f"""
                SELECT c.chunk_id, c.embed_text
                FROM chunks c
                LEFT JOIN {self.table} e USING (chunk_id)
                WHERE c.corpus_id = %s AND e.chunk_id IS NULL
                ORDER BY c.id
                """,
                (corpus_id,),
            ).fetchall()

        started = time.perf_counter()
        encoded = 0
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            vectors = self.encode_passages([r["embed_text"] for r in batch])
            with pool().connection() as conn, conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {self.table} (chunk_id, corpus_id, embedding) "
                    f"VALUES (%s, %s, %s)",
                    [
                        (row["chunk_id"], corpus_id, str(vector))
                        for row, vector in zip(batch, vectors, strict=True)
                    ],
                )
            encoded += len(batch)

        encode_seconds = time.perf_counter() - started
        return {
            "chunks_encoded": float(encoded),
            "encode_seconds": encode_seconds,
            "index_seconds": self.build_index(),
        }

    # -- querying ---------------------------------------------------------

    def table_exists(self) -> bool:
        with pool().connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM pg_class WHERE relname = %s AND relkind = 'r'",
                (self.table,),
            ).fetchone()
        return row is not None

    def assert_corpus_embedded(self, corpus_id: str) -> None:
        """An unembedded corpus is a defect, not a zero-recall measurement.

        Unlike BM25, a nearest-neighbour scan over a populated table always
        returns k rows, so an empty result here can only mean the corpus was
        never embedded for this model. Reporting that as "nothing relevant"
        would silently drop the corpus from the evaluation.
        """
        if not self.table_exists():
            raise IndexingError(
                f"embedding table {self.table} does not exist: corpus {corpus_id!r} "
                f"is not embedded for {self.embed_model}"
            )
        with pool().connection() as conn:
            row = conn.execute(
                f"SELECT count(*) AS n FROM {self.table} WHERE corpus_id = %s",
                (corpus_id,),
            ).fetchone()
        if int(row["n"]) == 0:
            raise IndexingError(
                f"corpus {corpus_id!r} is not embedded in {self.table}; "
                f"run the indexing step for {self.embed_model} before querying"
            )

    def search(self, query: str, corpus_id: str, k: int) -> list[Hit]:
        self.assert_corpus_embedded(corpus_id)
        vector = str(self.encode_query(query))
        with pool().connection() as conn:
            rows = conn.execute(
                f"""
                SELECT e.chunk_id, c.doc_id,
                       1 - (e.embedding <=> %s::vector) AS score
                FROM {self.table} e
                JOIN chunks c USING (chunk_id)
                WHERE e.corpus_id = %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (vector, corpus_id, vector, k),
            ).fetchall()
        return [
            Hit(chunk_id=r["chunk_id"], doc_id=r["doc_id"], score=float(r["score"]), rank=i + 1)
            for i, r in enumerate(rows)
        ]
