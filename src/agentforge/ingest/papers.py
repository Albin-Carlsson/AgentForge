"""Fetch, parse, chunk and index arXiv papers into Postgres.

This is the wiring between the pieces: arxiv.search -> download_pdf ->
parse_pdf -> chunk_sections -> documents/chunks rows.

Two details carry the design:

  * The token counter is the EMBEDDING MODEL'S OWN tokenizer, not a word count.
    Chunk size is a measured arm, and a counter that disagrees with the model
    would silently move every boundary -- 256 wordpiece tokens is roughly 190
    words, so a word-based count would produce chunks a third too large.
  * Every chunk's page and bbox come from the span map, so a citation resolves
    to a rectangle on a page rather than to a filename.

There is no partial-ingest mode. A paper that fails to parse aborts the run,
because indexing 97% of a corpus silently drops exactly the hard papers and
inflates every recall number computed over it.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from functools import lru_cache

from agentforge.config import IngestConfig
from agentforge.db.conn import pool
from agentforge.errors import IngestError
from agentforge.ingest import arxiv
from agentforge.ingest.chunker import chunk_sections
from agentforge.ingest.pdf import bbox_for, parse_pdf

CORPUS_ID = "arxiv-rag"
TIER = "arxiv40"

# Narrow on purpose: labels require understanding the documents, and the whole
# evaluation plan rests on being able to write 40 good questions in an afternoon.
QUERY = (
    "(cat:cs.IR OR cat:cs.CL) AND ("
    'abs:"retrieval-augmented" OR abs:"dense retrieval" OR abs:"reranking" OR '
    'abs:"hybrid retrieval" OR abs:"passage retrieval" OR abs:"RAG"'
    ")"
)


@dataclass(frozen=True)
class IngestReport:
    papers: int
    chunks: int
    pages: int
    published_from: str
    published_to: str
    mean_chunk_tokens: float
    skipped_duplicates: int


@lru_cache(maxsize=2)
def _tokenizer(embed_model: str):
    from sentence_transformers import SentenceTransformer

    from agentforge.settings import settings

    model = SentenceTransformer(embed_model, cache_folder=str(settings().model_dir))
    tokenizer = model.tokenizer
    # We are COUNTING tokens, not encoding for inference, so a section longer
    # than the model's 512-token window is expected and not a problem. Raising
    # the limit suppresses a per-section "sequence length is longer than the
    # specified maximum" warning that would otherwise bury the ingest output.
    tokenizer.model_max_length = int(1e9)
    return tokenizer


def token_counter(embed_model: str):
    """Count tokens the way the embedding model will."""
    tokenizer = _tokenizer(embed_model)

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count


def _register_corpus(config: IngestConfig, embed_model: str) -> None:
    with pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO corpora (corpus_id, kind, ingest_config)
            VALUES (%s, 'arxiv', %s)
            ON CONFLICT (corpus_id) DO UPDATE SET ingest_config = EXCLUDED.ingest_config
            """,
            (
                CORPUS_ID,
                json.dumps(
                    {
                        **config.model_dump(mode="json"),
                        "embed_model_for_tokenizer": embed_model,
                        "query": QUERY,
                    }
                ),
            ),
        )


def _already_ingested() -> set[str]:
    with pool().connection() as conn:
        return {
            row["doc_id"]
            for row in conn.execute(
                "SELECT doc_id FROM documents WHERE corpus_id = %s", (CORPUS_ID,)
            )
        }


def ingest(
    config: IngestConfig,
    n_papers: int,
    embed_model: str = "BAAI/bge-base-en-v1.5",
    pdf_dir: pathlib.Path = pathlib.Path("data/pdfs"),
    page_size: int = 50,
) -> IngestReport:
    """Ingest `n_papers` recent papers. Idempotent: existing papers are skipped."""
    if config.corpus_id != CORPUS_ID:
        raise IngestError(
            f"config targets corpus {config.corpus_id!r} but this ingest writes "
            f"{CORPUS_ID!r}; refusing to mix corpora in one table"
        )
    _register_corpus(config, embed_model)
    count = token_counter(embed_model)
    existing = _already_ingested()

    papers: list[arxiv.PaperMeta] = []
    start = 0
    while len(papers) < n_papers:
        page = arxiv.search(QUERY, max_results=min(page_size, n_papers - len(papers)), start=start)
        if not page:
            break
        papers.extend(page)
        start += len(page)

    total_chunks = total_pages = skipped = 0
    chunk_tokens: list[int] = []
    published: list[str] = []

    for paper in papers[:n_papers]:
        if paper.arxiv_id in existing:
            skipped += 1
            continue

        path = arxiv.download_pdf(paper, pdf_dir)
        parsed = parse_pdf(path)
        chunks = chunk_sections(
            parsed.sections,
            title=parsed.title or paper.title,
            token_counter=count,
            chunk_tokens=config.chunk_tokens,
            overlap_tokens=config.chunk_overlap,
            contextual_prefix=config.contextual_prefix,
            parent_section=config.parent_section,
        )
        if not chunks:
            raise IngestError(f"{paper.arxiv_id}: parsed but produced no chunks")

        rows = []
        for chunk in chunks:
            located = bbox_for(parsed.spans, chunk.char_start, chunk.char_end)
            page_no, bbox = located[0] if located else (None, None)
            rows.append(
                (
                    f"{paper.arxiv_id}:{chunk.ordinal}",
                    paper.arxiv_id,
                    CORPUS_ID,
                    chunk.ordinal,
                    chunk.section_path,
                    page_no,
                    chunk.char_start,
                    chunk.char_end,
                    list(bbox) if bbox else None,
                    chunk.text,
                    chunk.embed_text,
                    chunk.parent_text,
                    chunk.n_tokens,
                )
            )
            chunk_tokens.append(chunk.n_tokens)

        with pool().connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                  (doc_id, corpus_id, source_type, source_ref, title, published, meta)
                VALUES (%s,%s,'pdf',%s,%s,%s,%s)
                ON CONFLICT (doc_id) DO NOTHING
                """,
                (
                    paper.arxiv_id, CORPUS_ID, paper.pdf_url,
                    parsed.title or paper.title, paper.published,
                    json.dumps(
                        {
                            "authors": paper.authors,
                            "categories": paper.categories,
                            "primary_category": paper.primary_category,
                            "abs_url": paper.abs_url,
                            "html_url": paper.html_url,
                            "n_pages": parsed.n_pages,
                            "parser_version": parsed.parser_version,
                            "n_sections": len(parsed.sections),
                        }
                    ),
                ),
            )
            cur.executemany(
                """
                INSERT INTO chunks
                  (chunk_id, doc_id, corpus_id, ordinal, section_path, page,
                   char_start, char_end, bbox, text, embed_text, parent_text, n_tokens)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                rows,
            )

        total_chunks += len(rows)
        total_pages += parsed.n_pages
        published.append(str(paper.published))

    with pool().connection() as conn:
        conn.execute(
            """
            UPDATE corpora SET
              n_documents = (SELECT count(*) FROM documents WHERE corpus_id = %s),
              n_chunks    = (SELECT count(*) FROM chunks    WHERE corpus_id = %s)
            WHERE corpus_id = %s
            """,
            (CORPUS_ID, CORPUS_ID, CORPUS_ID),
        )

    return IngestReport(
        papers=len(published),
        chunks=total_chunks,
        pages=total_pages,
        # Contamination-freedom for the end-to-end tier rests on this range
        # post-dating the generator's training cutoff, so it is reported.
        published_from=min(published) if published else "-",
        published_to=max(published) if published else "-",
        mean_chunk_tokens=sum(chunk_tokens) / len(chunk_tokens) if chunk_tokens else 0.0,
        skipped_duplicates=skipped,
    )
