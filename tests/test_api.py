"""Tests for the HTTP surface.

A research run takes tens of seconds, so the API is job-based: POST returns an
id, an in-process task does the work, GET polls. No broker -- one Postgres table
and one asyncio task, which is the right size for this system.

The researcher is injected, so these tests need no API key and make no model
calls. What they pin is the contract and the failure behaviour: a failed job is
RECORDED as failed with its error, never reported as an empty success.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentforge.api.main import build_app


class FakeResearcher:
    def __init__(self, outcome=None, error: Exception | None = None) -> None:
        self.outcome = outcome
        self.error = error
        self.questions: list[str] = []

    def run(self, question: str):
        self.questions.append(question)
        if self.error is not None:
            raise self.error
        return self.outcome


def outcome_for(question: str):
    from agentforge.agents.research import ResearchOutcome
    from agentforge.agents.verify import ClaimVerdict
    from agentforge.llm import Usage

    verdict = ClaimVerdict(
        claim_idx=0, claim_text="Hybrid helps.", chunk_id="p1:0",
        quote="combines BM25 and dense scores", quote_found=True,
        reason=None, entailed="entailed",
    )
    return ResearchOutcome(
        question=question,
        subquestions=[question],
        claims=[{"claim": "Hybrid helps.", "citations": [
            {"chunk_id": "p1:0", "quote": "combines BM25 and dense scores"}]}],
        verdicts=[verdict],
        report="Hybrid helps. [^1]\n\n[^1]: p1:0",
        stage_latency_ms={"retrieve": 10.0, "synthesize": 100.0},
        usage=Usage(calls=1, input_tokens=100, output_tokens=20, cost_usd=0.0002),
        counts={"citations": 1, "verified": 1, "groundedness": 1.0},
        retrieved={question: ["p1:0"]},
    )


@pytest.fixture
def client(db):
    researcher = FakeResearcher()
    app = build_app(researcher_factory=lambda arm: researcher)
    with TestClient(app) as test_client:
        test_client.researcher = researcher  # type: ignore[attr-defined]
        yield test_client


def test_healthz_reports_the_engine_and_its_extensions(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "pg_search" in body["engine"]
    assert "vector" in body["engine"]


def test_posting_a_question_returns_a_job_id_immediately(client):
    client.researcher.outcome = outcome_for("Does hybrid help?")
    response = client.post("/research", json={"question": "Does hybrid help?"})
    assert response.status_code == 202
    assert response.json()["job_id"]
    # Queued, not done: the run happens after the response, never inside it.
    assert response.json()["status"] == "queued"


def test_polling_a_finished_job_returns_the_cited_report(client):
    client.researcher.outcome = outcome_for("Does hybrid help?")
    job_id = client.post("/research", json={"question": "Does hybrid help?"}).json()["job_id"]
    body = client.get(f"/research/{job_id}").json()
    assert body["status"] == "done"
    assert "Hybrid helps." in body["report_md"]
    assert body["counts"]["verified"] == 1
    assert body["stage_latency_ms"]["synthesize"] == 100.0


def test_a_failing_job_is_recorded_as_failed_with_its_error(client):
    """Never an empty success: a swallowed failure would be indistinguishable
    from a question the corpus genuinely could not answer."""
    client.researcher.error = RuntimeError("retriever exploded")
    job_id = client.post("/research", json={"question": "boom"}).json()["job_id"]
    body = client.get(f"/research/{job_id}").json()
    assert body["status"] == "failed"
    assert "retriever exploded" in body["error"]
    assert body["report_md"] is None


def test_an_unknown_job_is_404(client):
    assert client.get("/research/does-not-exist").status_code == 404


def test_an_empty_question_is_rejected(client):
    assert client.post("/research", json={"question": "   "}).status_code == 422


def test_the_arm_is_selectable_so_both_can_be_served(client):
    client.researcher.outcome = outcome_for("q")
    response = client.post("/research", json={"question": "q", "arm": "single_agent"})
    assert response.status_code == 202
    assert client.get(f"/research/{response.json()['job_id']}").json()["arm"] == "single_agent"


def test_an_unknown_arm_is_rejected(client):
    assert client.post("/research", json={"question": "q", "arm": "nonsense"}).status_code == 422


def test_results_endpoint_serves_the_generated_ablation_table(client, tmp_path, monkeypatch):
    from agentforge.api import main as api

    table = tmp_path / "retrieval-scifact.md"
    table.write_text("| Arm | nDCG@10 |\n|---|---|\n| Hybrid | 0.7389 |\n")
    monkeypatch.setattr(api, "RESULTS_DIR", tmp_path)
    body = client.get("/results/retrieval-scifact").json()
    assert "0.7389" in body["markdown"]


def test_a_missing_results_file_is_404_not_an_empty_table(client, tmp_path, monkeypatch):
    from agentforge.api import main as api

    monkeypatch.setattr(api, "RESULTS_DIR", tmp_path)
    assert client.get("/results/nope").status_code == 404
