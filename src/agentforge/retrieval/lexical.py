"""Lexical retrieval: real Okapi BM25 via pg_search.

The trap this module exists to avoid: Postgres `tsvector` + `ts_rank_cd` is NOT
BM25 -- different term-frequency saturation, different length normalisation, no
k1/b at all. Claiming "BM25" and shipping `ts_rank` is the kind of overstatement
that costs more in an interview than the keyword gains. This is Tantivy-backed
Okapi BM25 through ParadeDB's `pg_search`, in the same Postgres as pgvector, so
there is no second store and no sync path between the two indexes.

Why lexical retrieval earns its place on this corpus specifically: technical
papers are dense with rare exact identifiers -- "BEIR", "nDCG@10",
"bge-reranker", "SciFact" -- which is precisely where dense retrieval is weakest
and BM25 strongest. The hybrid gain here has a mechanism behind it.

Behaviour verified against pg_search 0.26.3 rather than against documentation:

  * `key_field` is deprecated and a no-op since 0.26.0, so it is not passed.
  * A relation may hold only ONE ParadeDB index, so changing the analyzer is a
    drop-and-recreate. That makes index building an EXPLICIT step: `search`
    refuses to run against an index whose analyzer differs from the run config,
    rather than silently rebuilding mid-ablation.
  * k1/b are not exposed as index options; see `bm25_parameters`.
  * `ORDER BY paradedb.score(id)` must be standalone -- wrapping it (e.g. in
    `round()`) silently drops the Top-K index scan and degrades to a full scan.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from psycopg import sql

from agentforge.config import RetrievalConfig
from agentforge.db.conn import pool
from agentforge.errors import IndexingError
from agentforge.retrieval.embed import Hit

INDEX_NAME = "chunks_bm25_idx"

_TEXT_FIELDS_RE = re.compile(r"text_fields='(?P<json>.*?)'(?:,|\))", re.DOTALL)


@dataclass(frozen=True)
class IndexState:
    rebuilt: bool
    seconds: float
    analyzer: dict[str, object]


def bm25_parameters() -> dict[str, float | str]:
    """The BM25 constants actually in force.

    Tantivy implements Okapi BM25 with the standard k1=1.2, b=0.75 and does not
    surface them as index options. They are DOCUMENTED CONSTANTS here, not tuned
    knobs, and `RetrievalConfig` deliberately has no field for them.
    """
    return {"k1": 1.2, "b": 0.75, "source": "tantivy-default", "tunable": "no"}


def analyzer_spec(config: RetrievalConfig) -> dict[str, object]:
    """The analyzer configuration for the text field. This is what is tunable."""
    tokenizer: dict[str, object] = {"type": "default"}
    if config.bm25_stemmer is not None:
        tokenizer["stemmer"] = config.bm25_stemmer
    if config.bm25_stopwords is not None:
        tokenizer["stopwords_language"] = config.bm25_stopwords
    return tokenizer


def _text_fields(config: RetrievalConfig) -> str:
    return json.dumps(
        {
            "text": {"tokenizer": analyzer_spec(config)},
            # keyword tokenizer: the corpus id is an exact-match filter, not
            # free text, and keeping it in the index means the filter is applied
            # inside the BM25 scan instead of as a post-filter.
            "corpus_id": {"tokenizer": {"type": "keyword"}},
        },
        sort_keys=True,
    )


def index_exists() -> bool:
    with pool().connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM pg_class WHERE relname = %s AND relkind = 'i'", (INDEX_NAME,)
        ).fetchone()
    return row is not None


def live_analyzer() -> dict[str, object]:
    """Read back the analyzer the live index was actually built with."""
    if not index_exists():
        raise IndexingError(f"BM25 index {INDEX_NAME} does not exist")
    with pool().connection() as conn:
        row = conn.execute(
            "SELECT pg_get_indexdef(c.oid) AS ddl FROM pg_class c WHERE c.relname = %s",
            (INDEX_NAME,),
        ).fetchone()
    match = _TEXT_FIELDS_RE.search(str(row["ddl"]))
    if match is None:
        raise IndexingError(
            f"cannot read text_fields from the live definition of {INDEX_NAME}; "
            f"definition was: {row['ddl']}"
        )
    fields = json.loads(match.group("json"))
    return fields["text"]["tokenizer"]


def drop_index() -> None:
    with pool().connection() as conn:
        conn.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")


def build_index(config: RetrievalConfig) -> IndexState:
    """Create the BM25 index for this config. Explicit, never implicit.

    Returns `rebuilt=False` when the live index already matches, so the harness
    does not silently pay for a rebuild it did not ask for.
    """
    desired = analyzer_spec(config)
    if index_exists() and live_analyzer() == desired:
        return IndexState(rebuilt=False, seconds=0.0, analyzer=desired)

    started = time.perf_counter()
    with pool().connection() as conn:
        # Only one ParadeDB index per relation, so an analyzer change is a
        # replace rather than an addition.
        conn.execute(f"DROP INDEX IF EXISTS {INDEX_NAME}")
        # Index storage parameters cannot be bound as query parameters, so the
        # analyzer JSON is rendered as a literal by psycopg rather than
        # interpolated by hand.
        conn.execute(
            sql.SQL(
                "CREATE INDEX {name} ON chunks USING bm25 (id, text, corpus_id) "
                "WITH (text_fields={fields})"
            ).format(
                name=sql.Identifier(INDEX_NAME),
                fields=sql.Literal(_text_fields(config)),
            )
        )
    return IndexState(
        rebuilt=True, seconds=time.perf_counter() - started, analyzer=desired
    )


def assert_index_matches(config: RetrievalConfig) -> None:
    if not index_exists():
        raise IndexingError(
            f"BM25 index {INDEX_NAME} does not exist; build it explicitly before querying"
        )
    live, desired = live_analyzer(), analyzer_spec(config)
    if live != desired:
        raise IndexingError(
            f"live BM25 analyzer does not match the run config: "
            f"index has {live}, config wants {desired}. "
            "Rebuild explicitly -- a silent rebuild would thrash between ablation arms."
        )


def assert_corpus_indexed(corpus_id: str) -> None:
    """An unindexed corpus is a defect, and must not read as 'nothing relevant'."""
    with pool().connection() as conn:
        row = conn.execute(
            "SELECT count(*) AS n FROM chunks WHERE corpus_id = %s", (corpus_id,)
        ).fetchone()
    if int(row["n"]) == 0:
        raise IndexingError(
            f"corpus {corpus_id!r} has no indexed rows in `chunks`. "
            "Refusing to report an empty retrieval as a zero-recall measurement."
        )


def search(query: str, corpus_id: str, config: RetrievalConfig, k: int) -> list[Hit]:
    """BM25 top-k. Returns [] when nothing matches -- that is a measurement.

    The query goes through `paradedb.match`, which treats input as terms rather
    than as Tantivy query syntax. Passing raw question text to the query parser
    breaks on punctuation like "nDCG@10" or a trailing "?", and a parse error
    returning nothing would quietly understate the lexical arm.
    """
    assert_index_matches(config)
    assert_corpus_indexed(corpus_id)

    with pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT c.chunk_id, c.doc_id, paradedb.score(c.id) AS score
            FROM chunks c
            WHERE c.id @@@ paradedb.match('text', %s)
              AND c.id @@@ paradedb.term('corpus_id', %s)
            ORDER BY paradedb.score(c.id) DESC
            LIMIT %s
            """,
            (query, corpus_id, k),
        ).fetchall()

    return [
        Hit(chunk_id=r["chunk_id"], doc_id=r["doc_id"], score=float(r["score"]), rank=i + 1)
        for i, r in enumerate(rows)
    ]
