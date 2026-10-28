"""Load the frozen arXiv question set into queries and span-level qrels.

The set lives in the repo as JSONL (`data/arxiv40.jsonl`) so that its content is
reviewable and versioned; the database copy is derived from it. Each row's gold
is a list of chunk ids -- every chunk of the gold paper(s) that states the
answer, not just the first one found, or a retriever returning a valid
restatement would be scored as a miss.

"Frozen" is enforced, not described: once any run exists on the tier, a load
that would change a question, its answer or its gold raises. Before the first
run the set is still a draft and reloading replaces it, which is what lets a
human review edit it.

The file's sha256 is stored on every query, so a run can be tied to the exact
question-set version it was scored against.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

from agentforge.db.conn import pool
from agentforge.errors import IngestError

DEFAULT_PATH = pathlib.Path("data/arxiv40.jsonl")
_FIELDS = {"query_id", "question", "gold_answer", "kind", "gold_chunks", "answer_keys"}
_KINDS = {"single", "multi"}


def _parse(path: pathlib.Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    rows: list[dict] = []
    for lineno, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            raise IngestError(f"{path}:{lineno}: blank line in question set")
        row = json.loads(line)
        if set(row) != _FIELDS:
            raise IngestError(f"{path}:{lineno}: fields {sorted(row)} != {sorted(_FIELDS)}")
        if row["kind"] not in _KINDS:
            raise IngestError(f"{path}:{lineno}: kind {row['kind']!r} not in {sorted(_KINDS)}")
        if not row["question"].strip() or not row["gold_answer"].strip():
            raise IngestError(f"{path}:{lineno}: empty question or gold_answer")
        if not row["gold_chunks"] or len(set(row["gold_chunks"])) != len(row["gold_chunks"]):
            raise IngestError(f"{path}:{lineno}: gold_chunks empty or duplicated")
        if not row["answer_keys"] or not all(k.strip() for k in row["answer_keys"]):
            raise IngestError(f"{path}:{lineno}: answer_keys empty")
        rows.append(row)
    ids = [r["query_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise IngestError(f"{path}: duplicate query_id")
    return rows, hashlib.sha256(raw).hexdigest()


def _assert_not_frozen_or_unchanged(conn, tier: str, rows: list[dict]) -> None:
    n_runs = conn.execute("SELECT count(*) AS n FROM runs WHERE tier = %s", (tier,)).fetchone()["n"]
    if n_runs == 0:
        return
    stored = {
        r["query_id"]: r
        for r in conn.execute(
            """
            SELECT q.query_id, q.text, q.gold_answer, q.meta->'answer_keys' AS keys,
                   array_agg(r.chunk_id ORDER BY r.chunk_id) AS gold
            FROM queries q JOIN qrels r USING (query_id)
            WHERE q.tier = %s GROUP BY q.query_id, q.text, q.gold_answer, q.meta
            """,
            (tier,),
        )
    }
    incoming = {r["query_id"]: r for r in rows}
    if set(stored) != set(incoming):
        raise IngestError(
            f"tier {tier!r} is frozen ({n_runs} runs exist) and the file adds or removes questions"
        )
    for qid, row in incoming.items():
        old = stored[qid]
        if (old["text"], old["gold_answer"], list(old["gold"]), old["keys"]) != (
            row["question"], row["gold_answer"], sorted(row["gold_chunks"]), row["answer_keys"]
        ):
            raise IngestError(f"tier {tier!r} is frozen ({n_runs} runs exist) and {qid} changed")


def load(corpus_id: str, tier: str, path: pathlib.Path = DEFAULT_PATH) -> dict[str, object]:
    """Validate the file against the indexed corpus and (re)load the tier."""
    rows, sha = _parse(path)
    wanted = {c for r in rows for c in r["gold_chunks"]}
    with pool().connection() as conn:
        found = {
            r["chunk_id"]
            for r in conn.execute(
                "SELECT chunk_id FROM chunks WHERE corpus_id = %s AND chunk_id = ANY(%s)",
                (corpus_id, list(wanted)),
            )
        }
        missing = sorted(wanted - found)
        if missing:
            raise IngestError(
                f"{len(missing)} gold chunks are not in corpus {corpus_id!r}, e.g. {missing[:3]}"
            )
        _assert_not_frozen_or_unchanged(conn, tier, rows)

        with conn.transaction(), conn.cursor() as cur:
            cur.execute("DELETE FROM queries WHERE tier = %s", (tier,))
            cur.executemany(
                """INSERT INTO queries (query_id, tier, corpus_id, text, gold_answer, meta)
                   VALUES (%s,%s,%s,%s,%s,%s)""",
                [
                    (
                        r["query_id"], tier, corpus_id, r["question"], r["gold_answer"],
                        json.dumps({
                            "kind": r["kind"], "answer_keys": r["answer_keys"],
                            "source_sha256": sha, "source": str(path),
                        }),
                    )
                    for r in rows
                ],
            )
            cur.executemany(
                "INSERT INTO qrels (query_id, chunk_id, relevance) VALUES (%s,%s,1)",
                [(r["query_id"], c) for r in rows for c in r["gold_chunks"]],
            )
    return {
        "questions": len(rows),
        "multi": sum(r["kind"] == "multi" for r in rows),
        "gold_chunks": len(wanted),
        "sha256": sha,
    }
