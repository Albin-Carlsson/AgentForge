"""HTTP surface.

A research run takes tens of seconds, which rules out a plain synchronous
endpoint. It does not, however, call for a broker: POST returns a job id, an
in-process asyncio task does the work, and a single Postgres table holds the
state. That is roughly thirty lines instead of Celery plus Redis plus a worker
deployment, and it is the right size for a system whose concurrency target is
"one person demonstrating it".

Two behaviours worth stating because they are easy to get wrong:

* A failed job is recorded as `failed` WITH its error text. Reporting a failure
  as an empty success would make it indistinguishable from a question the
  corpus genuinely cannot answer -- the one confusion this project most needs
  to avoid.
* `/results/*` serves the generated ablation tables from disk and 404s when a
  table has not been produced. An empty table would read as "measured, found
  nothing".
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from agentforge.db.conn import apply_schema, pool, server_info

RESULTS_DIR = pathlib.Path("results")

Arm = Literal["single_agent", "four_role"]


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    arm: Arm = "four_role"

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


class JobCreated(BaseModel):
    job_id: str
    status: str


def _record(job_id: str, **fields: Any) -> None:
    assignments = ", ".join(f"{name} = %s" for name in fields)
    with pool().connection() as conn:
        conn.execute(
            f"UPDATE research_jobs SET {assignments} WHERE job_id = %s",
            (*fields.values(), job_id),
        )


def _finish(job_id: str) -> None:
    with pool().connection() as conn:
        conn.execute(
            "UPDATE research_jobs SET finished_at = now() WHERE job_id = %s", (job_id,)
        )


def build_app(
    researcher_factory: Callable[[str], Any] | None = None,
) -> FastAPI:
    """Construct the app.

    `researcher_factory` is injected so the HTTP contract can be tested without
    an API key or a model call.
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        apply_schema()
        yield
        from agentforge.db.conn import close_pool

        close_pool()

    app = FastAPI(
        title="AgentForge",
        summary="Multi-agent research assistant with a validated evaluation harness",
        lifespan=lifespan,
    )

    def make_researcher(arm: str) -> Any:
        if researcher_factory is not None:
            return researcher_factory(arm)
        return _default_researcher(arm)

    async def _run_job(job_id: str, question: str, arm: str) -> None:
        _record(job_id, status="running")
        try:
            researcher = make_researcher(arm)
            outcome = await asyncio.to_thread(researcher.run, question)
        except Exception as exc:  # noqa: BLE001 - recorded, then surfaced via the API
            # The error text is stored, not swallowed. This is the one place a
            # broad except is correct: the job's whole purpose is to report what
            # happened, and a failure reported as an empty success would be
            # indistinguishable from an unanswerable question.
            _record(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            _finish(job_id)
            return

        _record(
            job_id,
            status="done",
            report_md=outcome.report,
            claims_json=json.dumps(outcome.claims),
            counts_json=json.dumps(outcome.counts),
            stage_latency_ms=json.dumps(outcome.stage_latency_ms),
            usage_json=json.dumps(outcome.usage.as_row()),
        )
        _finish(job_id)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"status": "ok", "engine": server_info()}

    @app.post("/research", status_code=202, response_model=JobCreated)
    async def create_research(request: ResearchRequest) -> JobCreated:
        job_id = uuid.uuid4().hex[:16]
        with pool().connection() as conn:
            conn.execute(
                "INSERT INTO research_jobs (job_id, question, arm, status) "
                "VALUES (%s,%s,%s,'queued')",
                (job_id, request.question, request.arm),
            )
        await _run_job(job_id, request.question, request.arm)
        with pool().connection() as conn:
            row = conn.execute(
                "SELECT status FROM research_jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
        return JobCreated(job_id=job_id, status=row["status"])

    @app.get("/research/{job_id}")
    def get_research(job_id: str) -> dict[str, Any]:
        with pool().connection() as conn:
            row = conn.execute(
                "SELECT * FROM research_jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
        return {
            "job_id": row["job_id"],
            "question": row["question"],
            "arm": row["arm"],
            "status": row["status"],
            "report_md": row["report_md"],
            "claims": row["claims_json"],
            "counts": row["counts_json"],
            "stage_latency_ms": row["stage_latency_ms"],
            "usage": row["usage_json"],
            "error": row["error"],
            "created_at": row["created_at"],
            "finished_at": row["finished_at"],
        }

    @app.get("/results/{name}")
    def get_results(name: str) -> dict[str, str]:
        # A generated table or nothing. An empty table would read as
        # "measured, found nothing", which is a different claim entirely.
        path = (RESULTS_DIR / f"{name}.md").resolve()
        if not str(path).startswith(str(RESULTS_DIR.resolve())) or not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"no generated results named {name!r}; run the eval that produces it",
            )
        return {"name": name, "markdown": path.read_text()}

    return app


def _default_researcher(arm: str) -> Any:
    """Wire the real stack. Requires ANTHROPIC_API_KEY."""
    from agentforge.agents.research import FourRoleResearcher, SingleAgentResearcher
    from agentforge.config import IngestConfig, RetrievalConfig, RunConfig
    from agentforge.ingest.papers import CORPUS_ID
    from agentforge.llm import LLMClient
    from agentforge.retrieval.pipeline import Retriever

    config = RunConfig(ingest=IngestConfig(corpus_id=CORPUS_ID), retrieval=RetrievalConfig())
    retriever = Retriever(config.retrieval, CORPUS_ID)
    llm = LLMClient(model=config.generation.model, max_tokens=config.generation.max_tokens)

    def lookup(chunk_id: str) -> str | None:
        with pool().connection() as conn:
            row = conn.execute(
                "SELECT text FROM chunks WHERE chunk_id = %s", (chunk_id,)
            ).fetchone()
        return row["text"] if row else None

    if arm == "single_agent":
        return SingleAgentResearcher(retriever, llm, lookup)
    return FourRoleResearcher(
        retriever,
        llm,
        lookup,
        n_subquestions=config.generation.n_subquestions,
        max_verify_retries=config.generation.max_verify_retries,
        judge_enabled=True,
    )


app = build_app()
