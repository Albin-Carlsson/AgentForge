"""BEIR SciFact loader -- the free retrieval tier.

This tier exists because the hand-written arXiv set cannot carry the statistical
weight the retrieval claims need. At N=300 with gold qrels, a 2-point nDCG delta
is real; at N=40 the confidence interval is roughly +/-7 points and reranking's
effect is undetectable. So parameter decisions get made here, where there is
power and where labels are free, and the arXiv tier is reserved for the
end-to-end numbers that get quoted.

SciFact specifically: ~5k scientific abstracts with claim-style queries. Short
documents, so chunking is a no-op and this tier can be stood up before the PDF
pipeline exists. That is what makes it milestone one.

Contamination is the known weakness -- these abstracts are in every model's
pretraining data -- which is exactly why nothing on this tier involves an LLM.
Ranks against qrels cannot be contaminated.
"""

from __future__ import annotations

import ir_datasets

from agentforge.db.conn import pool
from agentforge.errors import IngestError
from agentforge.ingest.normalize import canonical

DATASET = "beir/scifact/test"
CORPUS_ID = "scifact"
TIER = "scifact"


def _register_corpus(ingest_config: dict[str, object]) -> None:
    import json

    with pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO corpora (corpus_id, kind, ingest_config)
            VALUES (%s, 'beir', %s)
            ON CONFLICT (corpus_id) DO UPDATE SET ingest_config = EXCLUDED.ingest_config
            """,
            (CORPUS_ID, json.dumps(ingest_config)),
        )


def load(limit_docs: int | None = None) -> dict[str, int]:
    """Ingest SciFact documents, queries and qrels.

    One chunk per abstract: SciFact documents are already chunk-sized, and
    splitting them would fabricate a granularity mismatch with the
    document-level qrels.

    Fails loudly on any malformed record. A partial BEIR ingest would silently
    drop documents and inflate recall, since a missing document can never be a
    false positive.
    """
    dataset = ir_datasets.load(DATASET)

    _register_corpus(
        {
            "dataset": DATASET,
            "chunking": "one-chunk-per-abstract",
            "normalized": "canonical",
            "limit_docs": limit_docs,
        }
    )

    docs: list[tuple] = []
    chunks: list[tuple] = []
    for index, doc in enumerate(dataset.docs_iter()):
        if limit_docs is not None and index >= limit_docs:
            break
        title = (doc.title or "").strip()
        body = (doc.text or "").strip()
        if not body:
            raise IngestError(f"SciFact doc {doc.doc_id} has empty text")

        text = canonical(f"{title}\n\n{body}" if title else body)
        chunk_id = f"scifact:{doc.doc_id}:0"
        docs.append((doc.doc_id, CORPUS_ID, "beir", f"{DATASET}#{doc.doc_id}", title or None))
        chunks.append(
            (
                chunk_id, doc.doc_id, CORPUS_ID, 0,
                None, None, None, None,
                text,
                # embed_text carries the contextual prefix; `text` stays verbatim
                # because it is what a citation quote must match against.
                canonical(f"{title}. {body}") if title else text,
                None,
                len(text.split()),
            )
        )

    queries: list[tuple] = []
    for query in dataset.queries_iter():
        queries.append((f"scifact:{query.query_id}", TIER, CORPUS_ID, canonical(query.text)))

    known_docs = {d[0] for d in docs}
    qrels: list[tuple] = []
    for qrel in dataset.qrels_iter():
        if qrel.doc_id not in known_docs:
            if limit_docs is not None:
                continue  # truncated corpus: the qrel's document was not loaded
            raise IngestError(
                f"qrel references unknown doc {qrel.doc_id}; corpus and qrels disagree"
            )
        qrels.append((f"scifact:{qrel.query_id}", qrel.doc_id, int(qrel.relevance)))

    with pool().connection() as conn, conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO documents (doc_id, corpus_id, source_type, source_ref, title)
               VALUES (%s,%s,%s,%s,%s) ON CONFLICT (doc_id) DO NOTHING""",
            docs,
        )
        cur.executemany(
            """INSERT INTO chunks (chunk_id, doc_id, corpus_id, ordinal, section_path,
                                   page, char_start, char_end, text, embed_text,
                                   parent_text, n_tokens)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (chunk_id) DO NOTHING""",
            chunks,
        )
        cur.executemany(
            """INSERT INTO queries (query_id, tier, corpus_id, text)
               VALUES (%s,%s,%s,%s) ON CONFLICT (query_id) DO NOTHING""",
            queries,
        )
        # qrels has no natural primary key; clear this corpus's rows and reload
        # rather than accumulating duplicates across runs.
        cur.execute(
            "DELETE FROM qrels WHERE query_id IN (SELECT query_id FROM queries WHERE tier = %s)",
            (TIER,),
        )
        cur.executemany(
            "INSERT INTO qrels (query_id, doc_id, relevance) VALUES (%s,%s,%s)", qrels
        )
        cur.execute(
            """UPDATE corpora SET n_documents = %s, n_chunks = %s WHERE corpus_id = %s""",
            (len(docs), len(chunks), CORPUS_ID),
        )

    # A query with no positive judgement cannot be scored, and silently averaging
    # it in as 0.0 would understate every arm equally but wrongly.
    with pool().connection() as conn:
        unscorable = conn.execute(
            """
            SELECT count(*) AS n FROM queries q
            WHERE q.tier = %s AND NOT EXISTS (
                SELECT 1 FROM qrels r WHERE r.query_id = q.query_id AND r.relevance > 0
            )
            """,
            (TIER,),
        ).fetchone()

    return {
        "documents": len(docs),
        "chunks": len(chunks),
        "queries": len(queries),
        "qrels": len(qrels),
        "queries_without_positive_qrel": int(unscorable["n"]),
    }
