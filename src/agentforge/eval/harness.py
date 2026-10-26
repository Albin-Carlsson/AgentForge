"""The experiment harness: configs in, attributable numbers out.

Three rules are enforced here rather than left to operator discipline, because
operator discipline is what fails at 11pm on the fifth arm:

  1. A run stores the full config that produced it, plus the code revision.
     Without that, "+6 nDCG from reranking" is an anecdote.
  2. Results are stored per question. Aggregates are a GROUP BY; the per-question
     rows are what let you find the handful of questions that regressed, which
     is where the insight is.
  3. Comparisons are paired and single-knob. `compare_runs` refuses anything
     else, so an unattributable delta cannot reach a report.

Scoring granularity is read from the qrels, never assumed: BEIR judges whole
documents, the arXiv tier judges spans. Guessing wrong silently changes every
recall number.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from agentforge.config import RunConfig, assert_comparable, git_sha
from agentforge.db.conn import pool
from agentforge.errors import MeasurementError
from agentforge.eval.metrics_retrieval import ndcg_at_k, recall_at_k, reciprocal_rank
from agentforge.eval.stats import PairedDelta, paired_bootstrap, percentile
from agentforge.retrieval.pipeline import Retriever

_METRICS = ("recall", "ndcg", "mrr")


@dataclass(frozen=True)
class RunHandle:
    run_id: str
    config_hash: str
    n_queries: int


@dataclass(frozen=True)
class Comparison:
    knob: str
    metric: str
    baseline_run: str
    treatment_run: str
    delta: PairedDelta
    same_git_sha: bool

    def describe(self) -> str:
        caveat = "" if self.same_git_sha else "  [differing git SHAs]"
        return f"{self.knob}: {self.delta.describe(self.metric)}{caveat}"


# ------------------------------------------------------------------ execution


def execute_retrieval_run(
    config: RunConfig,
    tier: str,
    arm_label: str,
    cache_bypass: bool = False,
    notes: str | None = None,
) -> RunHandle:
    """Run one retrieval config over every query in a tier.

    Stores every stage's ranking, so a single pass yields the diagnostics for
    "what did the reranker move?" as well as the final ranking the metrics use.
    """
    run_id = f"{tier}-{arm_label}-{config.config_hash}-{uuid.uuid4().hex[:6]}"

    with pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO runs (run_id, config_hash, config_json, git_sha, arm_label,
                              tier, cache_bypass, notes)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                run_id, config.config_hash, config.canonical(), git_sha(),
                arm_label, tier, cache_bypass, notes,
            ),
        )
        queries = conn.execute(
            "SELECT query_id, text FROM queries WHERE tier = %s ORDER BY query_id",
            (tier,),
        ).fetchall()

    if not queries:
        raise MeasurementError(f"tier {tier!r} has no queries")

    retriever = Retriever(config.retrieval, config.ingest.corpus_id)
    rows: list[tuple] = []
    latencies: list[tuple[str, float]] = []

    for query in queries:
        trace = retriever.retrieve(query["text"])
        latencies.append((query["query_id"], trace.total_ms))
        for stage, hits in trace.stages.items():
            for hit in hits:
                rows.append(
                    (run_id, query["query_id"], stage, hit.rank,
                     hit.chunk_id, hit.doc_id, hit.score)
                )

    with pool().connection() as conn, conn.cursor() as cur:
        if rows:
            cur.executemany(
                """INSERT INTO retrieval_results
                   (run_id, query_id, stage, rank, chunk_id, doc_id, score)
                   VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                rows,
            )
        # Retrieval-only runs have no answer rows, so stage latency is parked on
        # the run's notes field as structured JSON rather than inventing a table.
        cur.execute(
            "UPDATE runs SET finished_at = now(), notes = %s WHERE run_id = %s",
            (json.dumps({"latency_ms": dict(latencies), "notes": notes}), run_id),
        )

    return RunHandle(run_id=run_id, config_hash=config.config_hash, n_queries=len(queries))


# -------------------------------------------------------------------- loading


def load_run(run_id: str) -> dict:
    with pool().connection() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,)).fetchone()
    if row is None:
        raise MeasurementError(f"no such run: {run_id}")
    return dict(row)


def load_retrieval_rows(run_id: str, stage: str | None = None) -> list[dict]:
    query = "SELECT * FROM retrieval_results WHERE run_id = %s"
    params: tuple = (run_id,)
    if stage is not None:
        query += " AND stage = %s"
        params += (stage,)
    with pool().connection() as conn:
        return [dict(r) for r in conn.execute(query + " ORDER BY query_id, rank", params)]


def run_config(run_id: str) -> RunConfig:
    """Rehydrate the frozen config a run was executed with."""
    stored = load_run(run_id)["config_json"]
    raw = stored if isinstance(stored, str) else json.dumps(stored)
    return RunConfig.model_validate_json(raw)


def scoring_granularity(tier: str) -> str:
    """Read from the qrels: 'doc_id' for BEIR, 'chunk_id' for span-level labels."""
    with pool().connection() as conn:
        row = conn.execute(
            """
            SELECT count(*) FILTER (WHERE r.doc_id IS NOT NULL)   AS docs,
                   count(*) FILTER (WHERE r.chunk_id IS NOT NULL) AS chunks
            FROM qrels r JOIN queries q USING (query_id)
            WHERE q.tier = %s
            """,
            (tier,),
        ).fetchone()
    docs, chunks = int(row["docs"]), int(row["chunks"])
    if docs and chunks:
        raise MeasurementError(
            f"tier {tier!r} mixes document-level and span-level qrels; "
            "scoring granularity would be ambiguous"
        )
    if docs:
        return "doc_id"
    if chunks:
        return "chunk_id"
    raise MeasurementError(f"tier {tier!r} has no qrels to score against")


def _final_stage(run_id: str) -> str:
    with pool().connection() as conn:
        stages = {
            r["stage"]
            for r in conn.execute(
                "SELECT DISTINCT stage FROM retrieval_results WHERE run_id = %s", (run_id,)
            )
        }
    for stage in ("reranked", "fused", "dense", "bm25"):
        if stage in stages:
            return stage
    raise MeasurementError(f"run {run_id} stored no retrieval stages")


def _gains(tier: str, granularity: str) -> dict[str, dict[str, int]]:
    with pool().connection() as conn:
        rows = conn.execute(
            f"""
            SELECT q.query_id, r.{granularity} AS key, r.relevance
            FROM qrels r JOIN queries q USING (query_id)
            WHERE q.tier = %s AND r.{granularity} IS NOT NULL
            """,
            (tier,),
        ).fetchall()
    gains: dict[str, dict[str, int]] = {}
    for row in rows:
        gains.setdefault(row["query_id"], {})[row["key"]] = int(row["relevance"])
    return gains


def _retrieved(run_id: str, stage: str, granularity: str) -> dict[str, list[str]]:
    """Ranked keys per query, de-duplicated at the scoring granularity.

    Two chunks of the same document occupy one document slot, so collapsing them
    is required for document-level recall to mean what it says.
    """
    out: dict[str, list[str]] = {}
    for row in load_retrieval_rows(run_id, stage=stage):
        keys = out.setdefault(row["query_id"], [])
        key = row[granularity]
        if key not in keys:
            keys.append(key)
    return out


# -------------------------------------------------------------------- scoring


def per_query_metric(run_id: str, metric: str, k: int) -> dict[str, float]:
    if metric not in _METRICS:
        raise MeasurementError(f"unknown metric {metric!r}; expected one of {_METRICS}")

    run = load_run(run_id)
    final_k = run_config(run_id).retrieval.final_k
    if k > final_k:
        raise MeasurementError(
            f"cannot score {metric}@{k} on run {run_id}: its config caps retrieval at "
            f"final_k={final_k}, so every arm is structurally limited to @{final_k}. "
            f"Score at k<={final_k}, or re-run with a larger final_k."
        )
    granularity = scoring_granularity(run["tier"])
    gains = _gains(run["tier"], granularity)
    retrieved = _retrieved(run_id, _final_stage(run_id), granularity)

    scores: dict[str, float] = {}
    for query_id, query_gains in gains.items():
        # A query the retriever returned nothing for scores zero. That is the
        # honest result, and dropping it would inflate the mean.
        ranked = retrieved.get(query_id, [])
        relevant = {key for key, gain in query_gains.items() if gain > 0}
        if not relevant:
            raise MeasurementError(
                f"query {query_id} has no positive qrel and cannot be scored"
            )
        if metric == "recall":
            scores[query_id] = recall_at_k(ranked, relevant, k)
        elif metric == "ndcg":
            scores[query_id] = ndcg_at_k(ranked, query_gains, k)
        else:
            scores[query_id] = reciprocal_rank(ranked, relevant)
    return scores


def score_run(run_id: str, k: int) -> dict[str, float | int | str]:
    recall = per_query_metric(run_id, "recall", k)
    ndcg = per_query_metric(run_id, "ndcg", k)
    mrr = per_query_metric(run_id, "mrr", k)
    n = len(recall)
    return {
        f"recall@{k}": sum(recall.values()) / n,
        f"nDCG@{k}": sum(ndcg.values()) / n,
        "MRR": sum(mrr.values()) / n,
        "n_queries": n,
        "stage": _final_stage(run_id),
    }


def latency_percentile(run_id: str, q: float) -> float:
    """Latency is only honest from a cache-bypassed run."""
    run = load_run(run_id)
    if not run["cache_bypass"]:
        raise MeasurementError(
            f"run {run_id} was not cache-bypassed, so its timings include cache hits; "
            "latency percentiles from it would be fiction"
        )
    notes = run["notes"]
    payload = json.loads(notes) if isinstance(notes, str) else (notes or {})
    samples = list((payload.get("latency_ms") or {}).values())
    if not samples:
        raise MeasurementError(f"run {run_id} recorded no latency samples")
    return percentile([float(s) for s in samples], q)


def compare_runs(baseline_run: str, treatment_run: str, metric: str, k: int) -> Comparison:
    """Paired, single-knob comparison. Refuses anything else."""
    base, treat = load_run(baseline_run), load_run(treatment_run)
    if base["tier"] != treat["tier"]:
        raise MeasurementError(
            f"runs are on different tiers ({base['tier']} vs {treat['tier']}): "
            "a comparison across question sets is not paired"
        )
    knob = assert_comparable(run_config(baseline_run), run_config(treatment_run))
    return Comparison(
        knob=knob,
        metric=f"{metric}@{k}",
        baseline_run=baseline_run,
        treatment_run=treatment_run,
        delta=paired_bootstrap(
            per_query_metric(baseline_run, metric, k),
            per_query_metric(treatment_run, metric, k),
        ),
        same_git_sha=base["git_sha"] == treat["git_sha"],
    )
