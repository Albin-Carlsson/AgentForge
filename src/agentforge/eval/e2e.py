"""End-to-end runs: the three research arms over the frozen arXiv questions.

Every per-question output is stored -- the answer, its token accounting, and one
row per citation verdict -- and every metric is computed from those rows
afterwards, so a scoring change never needs a re-run.

**Resumable by construction.** A run that stops part-way (an API error, an
interruption) is left unfinished, and scoring refuses unfinished runs.
Re-running the same config replays every completed call from the cache at no
cost and carries on. No question is ever skipped to make a run complete.

**Cost is reported at list price as if uncached.** The cache makes a re-run
free, so "what this run billed" depends on what ran before it. What an answer
costs the system is the tokens it consumed, priced at list; that is what the
`cost_usd` column holds, and it is identical on a first run and a replay.

**Latency is not reported from these runs.** They run through the cache, so a
replayed call's wall-clock is SQL, not model time. The answers table still
records stage timings, flagged by the run's `cache_bypass` being false, which
the latency query refuses.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from agentforge.agents.research import ControlAnswerer, FourRoleResearcher, SingleAgentResearcher
from agentforge.config import RunConfig, git_sha
from agentforge.db.conn import pool
from agentforge.errors import MeasurementError
from agentforge.eval.answer_keys import key_recall
from agentforge.eval.harness import RunHandle
from agentforge.eval.stats import PairedDelta, paired_bootstrap
from agentforge.llm import generator_client, judge_client, price
from agentforge.obs import RecordingObserver
from agentforge.retrieval.pipeline import Retriever


def _lookup(chunk_id: str) -> str | None:
    with pool().connection() as conn:
        row = conn.execute("SELECT text FROM chunks WHERE chunk_id = %s", (chunk_id,)).fetchone()
    return row["text"] if row else None


def build_arm(config: RunConfig, observer: RecordingObserver, transport: Any = None) -> Any:
    """`transport` replaces the provider transport for both clients (tests)."""
    generation = config.generation
    llm = generator_client(generation, observer=observer, transport=transport)
    if generation.graph == "none":
        return ControlAnswerer(llm)
    judge = judge_client(config.judge, observer=observer, transport=transport)
    retriever = Retriever(config.retrieval, config.ingest.corpus_id)
    if generation.graph == "single_agent":
        return SingleAgentResearcher(retriever, llm, _lookup, judge_llm=judge)
    return FourRoleResearcher(
        retriever, llm, _lookup,
        n_subquestions=generation.n_subquestions,
        max_verify_retries=generation.max_verify_retries,
        judge_llm=judge,
    )


def execute_e2e_run(
    config: RunConfig, tier: str, arm_label: str, transport: Any = None
) -> RunHandle:
    run_id = f"{tier}-{arm_label}-{config.config_hash}-{uuid.uuid4().hex[:6]}"
    with pool().connection() as conn:
        conn.execute(
            """INSERT INTO runs (run_id, config_hash, config_json, git_sha, arm_label, tier)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (run_id, config.config_hash, config.canonical(), git_sha(), arm_label, tier),
        )
        queries = conn.execute(
            "SELECT query_id, text FROM queries WHERE tier = %s ORDER BY query_id", (tier,)
        ).fetchall()
    if not queries:
        raise MeasurementError(f"tier {tier!r} has no queries")

    observer = RecordingObserver()
    arm = build_arm(config, observer, transport)

    for index, query in enumerate(queries, start=1):
        first_call = len(observer.generations)
        outcome = arm.run(query["text"])
        calls = observer.generations[first_call:]
        input_tokens = sum(c["input_tokens"] for c in calls)
        output_tokens = sum(c["output_tokens"] for c in calls)
        list_cost = sum(price(c["model"], c["input_tokens"], c["output_tokens"]) for c in calls)
        hits = sum(c["cached"] for c in calls)

        with pool().connection() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO answers (run_id, query_id, answer_md, stage_latency_ms, latency_ms,
                                        input_tokens, output_tokens, cost_usd)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    run_id, query["query_id"], outcome.report,
                    json.dumps(outcome.stage_latency_ms | {"llm_calls": len(calls)}),
                    int(sum(v for k, v in outcome.stage_latency_ms.items() if k != "research_serial_sum")),
                    input_tokens, output_tokens, list_cost,
                ),
            )
            cur.executemany(
                """INSERT INTO claims (run_id, query_id, claim_idx, claim_text, chunk_id, quote,
                                       quote_found, entailed, judge_model, verify_pass)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                [
                    (
                        run_id, query["query_id"], position, v.claim_text, v.chunk_id, v.quote,
                        v.quote_found, v.entailed,
                        config.judge.model if v.entailed is not None else None, v.verify_pass,
                    )
                    # claim_idx in the table is per-citation: one claim can carry
                    # several citations, and each gets its own verdict row.
                    for position, v in enumerate(outcome.verdicts)
                ],
            )
        print(
            f"  [{index:>2}/{len(queries)}] {query['query_id']}: {len(calls)} calls "
            f"({hits} cached), {input_tokens + output_tokens} tokens",
            flush=True,
        )

    with pool().connection() as conn:
        conn.execute("UPDATE runs SET finished_at = now() WHERE run_id = %s", (run_id,))
    return RunHandle(run_id=run_id, config_hash=config.config_hash, n_queries=len(queries))


# --------------------------------------------------------------------- scoring


@dataclass(frozen=True)
class E2EScores:
    run_id: str
    arm_label: str
    n: int
    correctness: float
    citations: int
    groundedness: float | None
    fabrication_rate: float | None
    unsupported_rate: float | None
    citation_doc_precision: float | None
    gold_chunk_cited: float | None
    tokens_per_question: float
    list_cost_usd: float


def per_question_correctness(run_id: str) -> dict[str, float]:
    with pool().connection() as conn:
        run = conn.execute("SELECT finished_at FROM runs WHERE run_id = %s", (run_id,)).fetchone()
        if run is None or run["finished_at"] is None:
            raise MeasurementError(
                f"run {run_id} is unfinished; re-run it (completed calls replay from the "
                "cache) rather than scoring a partial question set"
            )
        rows = conn.execute(
            """SELECT a.query_id, a.answer_md, q.meta->'answer_keys' AS keys
               FROM answers a JOIN queries q USING (query_id) WHERE a.run_id = %s""",
            (run_id,),
        ).fetchall()
    return {r["query_id"]: key_recall(r["answer_md"], list(r["keys"])) for r in rows}


def score_e2e_run(run_id: str) -> E2EScores:
    correctness = per_question_correctness(run_id)
    with pool().connection() as conn:
        arm = conn.execute("SELECT arm_label FROM runs WHERE run_id = %s", (run_id,)).fetchone()
        claims = conn.execute(
            """SELECT c.query_id, c.chunk_id, c.quote_found, c.entailed, ch.doc_id
               FROM claims c LEFT JOIN chunks ch ON ch.chunk_id = c.chunk_id
               WHERE c.run_id = %s""",
            (run_id,),
        ).fetchall()
        gold = conn.execute(
            """SELECT r.query_id, r.chunk_id, ch.doc_id FROM qrels r
               JOIN chunks ch ON ch.chunk_id = r.chunk_id
               WHERE r.query_id IN (SELECT query_id FROM answers WHERE run_id = %s)""",
            (run_id,),
        ).fetchall()
        totals = conn.execute(
            "SELECT sum(input_tokens + output_tokens) AS t, sum(cost_usd) AS c, count(*) AS n "
            "FROM answers WHERE run_id = %s",
            (run_id,),
        ).fetchone()

    gold_docs: dict[str, set[str]] = {}
    gold_chunks: dict[str, set[str]] = {}
    for g in gold:
        gold_docs.setdefault(g["query_id"], set()).add(g["doc_id"])
        gold_chunks.setdefault(g["query_id"], set()).add(g["chunk_id"])

    n_cit = len(claims)
    if n_cit:
        found = [c for c in claims if c["quote_found"]]
        unjudged = [c for c in found if c["entailed"] is None]
        if unjudged:
            raise MeasurementError(f"run {run_id}: {len(unjudged)} found quotes were never judged")
        groundedness = sum(c["entailed"] == "entailed" for c in found) / n_cit
        fabrication = sum(not c["quote_found"] for c in claims) / n_cit
        unsupported = (
            sum(c["entailed"] in ("unsupported", "contradicted") for c in found) / len(found)
            if found else None
        )
        resolvable = [c for c in claims if c["doc_id"] is not None]
        doc_precision = (
            sum(c["doc_id"] in gold_docs[c["query_id"]] for c in resolvable) / len(resolvable)
            if resolvable else None
        )
        cited_by_q: dict[str, set[str]] = {}
        for c in found:
            cited_by_q.setdefault(c["query_id"], set()).add(c["chunk_id"])
        gold_cited = sum(
            bool(cited_by_q.get(q, set()) & gold_chunks[q]) for q in correctness
        ) / len(correctness)
    else:
        groundedness = fabrication = unsupported = doc_precision = gold_cited = None

    return E2EScores(
        run_id=run_id,
        arm_label=arm["arm_label"],
        n=len(correctness),
        correctness=sum(correctness.values()) / len(correctness),
        citations=n_cit,
        groundedness=groundedness,
        fabrication_rate=fabrication,
        unsupported_rate=unsupported,
        citation_doc_precision=doc_precision,
        gold_chunk_cited=gold_cited,
        tokens_per_question=float(totals["t"]) / totals["n"],
        list_cost_usd=float(totals["c"]),
    )


def compare_correctness(baseline_run: str, treatment_run: str) -> PairedDelta:
    return paired_bootstrap(
        per_question_correctness(baseline_run), per_question_correctness(treatment_run)
    )
