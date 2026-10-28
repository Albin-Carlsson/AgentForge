"""Tests for the two research agents.

Both arms are exercised with a scripted transport and a stub retriever, so the
orchestration is tested without API calls. What matters here is not that a model
answers well, but that the graph's structural guarantees hold -- they are the
justification for the word "multi-agent" being on a resume at all:

  * Researchers are CONTEXT-ISOLATED: each sees only its own sub-question's
    evidence, never the others'.
  * The Verifier never sees the Synthesizer's reasoning -- only a claim and a
    quote.
  * The verify loop is BOUNDED at one retry, and a claim still rejected ships
    flagged rather than dropped or retried forever.
  * The single-agent baseline is a real arm, not a straw man.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentforge.agents.research import FourRoleResearcher, SingleAgentResearcher
from agentforge.errors import ConfigError
from agentforge.llm import LLMClient
from agentforge.retrieval.embed import Hit

CHUNKS = {
    "p1:0": "Hybrid retrieval combines BM25 and dense scores, evaluated on SciFact.",
    "p2:0": "A cross-encoder rescores candidates jointly with the query.",
    "p3:0": "Bi-encoders embed queries and passages separately and are fast.",
}


class StubRetriever:
    """Returns a fixed ranking and records every query it was asked."""

    def __init__(self, hits: list[str] | None = None) -> None:
        self.queries: list[str] = []
        self.hit_ids = hits or list(CHUNKS)

    def retrieve(self, query: str):
        self.queries.append(query)

        hits = [
            Hit(chunk_id=c, doc_id=c.split(":")[0], score=1.0 / (i + 1), rank=i + 1)
            for i, c in enumerate(self.hit_ids)
        ]
        return SimpleNamespace(
            stages={"fused": hits},
            latency_ms={"bm25": 1.0, "dense": 2.0},
            final_stage="fused",
            final=hits,
            total_ms=3.0,
        )


class ScriptedTransport:
    """Replays replies in order and records the system prompt of each call."""

    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def __call__(self, *, model, system, messages, max_tokens, schema, thinking):
        self.calls.append({"system": system, "prompt": messages[0]["content"], "schema": schema})
        if not self.replies:
            raise AssertionError(f"unexpected extra LLM call: {messages[0]['content'][:80]}")
        parsed = self.replies.pop(0)
        return {"parsed": parsed, "input_tokens": 100, "output_tokens": 20}


def claims(*specs: tuple[str, str, str]) -> dict:
    return {
        "claims": [
            {"claim": c, "citations": [{"chunk_id": cid, "quote": q}]} for c, cid, q in specs
        ]
    }


def llm(transport, **kwargs) -> LLMClient:
    return LLMClient(model="claude-haiku-4-5", transport=transport, cache_bypass=True, **kwargs)


# ------------------------------------------------------------- single agent


def test_single_agent_makes_exactly_one_retrieval_and_one_generation(db):
    retriever = StubRetriever()
    transport = ScriptedTransport(
        [claims(("Hybrid helps.", "p1:0", "combines BM25 and dense scores"))]
    )
    outcome = SingleAgentResearcher(retriever, llm(transport), CHUNKS.get).run("Does hybrid help?")

    assert retriever.queries == ["Does hybrid help?"]
    assert len(transport.calls) == 1
    assert outcome.subquestions == ["Does hybrid help?"]
    assert len(outcome.claims) == 1
    assert outcome.verdicts[0].quote_found is True


def test_single_agent_verifies_quotes_deterministically(db):
    transport = ScriptedTransport(
        [claims(("Hybrid helps.", "p1:0", "a quote that is simply not present"))]
    )
    outcome = SingleAgentResearcher(StubRetriever(), llm(transport), CHUNKS.get).run("q")
    assert outcome.verdicts[0].quote_found is False
    assert outcome.counts["fabricated_quotes"] == 1


def test_single_agent_reports_stage_latency(db):
    transport = ScriptedTransport([claims(("A.", "p1:0", "combines BM25 and dense scores"))])
    outcome = SingleAgentResearcher(StubRetriever(), llm(transport), CHUNKS.get).run("q")
    assert "retrieve" in outcome.stage_latency_ms
    assert "synthesize" in outcome.stage_latency_ms


# --------------------------------------------------------------- four-role


def test_four_role_plans_then_researches_each_subquestion(db):
    retriever = StubRetriever()
    transport = ScriptedTransport(
        [
            {"subquestions": ["What is hybrid retrieval?", "Does reranking help?"]},
            claims(("Hybrid combines two retrievers.", "p1:0", "combines BM25 and dense scores")),
            claims(("A cross-encoder rescores jointly.", "p2:0", "rescores candidates jointly with the query")),
        ]
    )
    outcome = FourRoleResearcher(retriever, llm(transport), CHUNKS.get).run("Explain hybrid.")

    assert outcome.subquestions == ["What is hybrid retrieval?", "Does reranking help?"]
    # One retrieval per sub-question, using the sub-question as the query.
    assert retriever.queries == ["What is hybrid retrieval?", "Does reranking help?"]
    assert len(outcome.claims) == 2


def test_researchers_are_context_isolated(db):
    """The measurable argument for more than one agent.

    Each Researcher's prompt must contain only its own sub-question. If one
    prompt carried both, this would be a single agent wearing a hat.
    """
    transport = ScriptedTransport(
        [
            {"subquestions": ["Sub one about hybrid", "Sub two about reranking"]},
            claims(("A.", "p1:0", "combines BM25 and dense scores")),
            claims(("B.", "p2:0", "rescores candidates jointly with the query")),
        ]
    )
    FourRoleResearcher(StubRetriever(), llm(transport), CHUNKS.get).run("Explain.")

    researcher_prompts = [c["prompt"] for c in transport.calls[1:]]
    assert len(researcher_prompts) == 2
    assert "Sub one about hybrid" in researcher_prompts[0]
    assert "Sub two about reranking" not in researcher_prompts[0]
    assert "Sub two about reranking" in researcher_prompts[1]
    assert "Sub one about hybrid" not in researcher_prompts[1]


def test_the_verifier_never_sees_the_generators_reasoning(db):
    """Adversarial separation: the judge gets a claim and a quote, nothing else."""
    transport = ScriptedTransport(
        [
            {"subquestions": ["Sub one"]},
            claims(("Hybrid helps.", "p1:0", "combines BM25 and dense scores")),
            {"verdict": "entailed"},
        ]
    )
    shared = llm(transport)
    FourRoleResearcher(
        StubRetriever(), shared, CHUNKS.get, judge_llm=shared
    ).run("Explain hybrid retrieval please.")

    judge_call = transport.calls[-1]
    assert "Explain hybrid retrieval please." not in judge_call["prompt"]
    assert "Sub one" not in judge_call["prompt"]
    assert "Hybrid helps." in judge_call["prompt"]
    assert "combines BM25 and dense scores" in judge_call["prompt"]


def test_a_rejected_claim_triggers_exactly_one_retry(db):
    transport = ScriptedTransport(
        [
            {"subquestions": ["Sub one"]},
            claims(("Hybrid helps.", "p1:0", "not in the chunk at all")),   # fabricated
            claims(("Hybrid helps.", "p1:0", "combines BM25 and dense scores")),  # retry
        ]
    )
    outcome = FourRoleResearcher(StubRetriever(), llm(transport), CHUNKS.get).run("q")
    assert len(transport.calls) == 3, "plan, research, one retry"
    assert outcome.verdicts[0].quote_found is True
    assert outcome.verdicts[0].verify_pass == 1


def test_a_claim_rejected_twice_ships_flagged_rather_than_looping(db):
    transport = ScriptedTransport(
        [
            {"subquestions": ["Sub one"]},
            claims(("Hybrid helps.", "p1:0", "still not present")),
            claims(("Hybrid helps.", "p1:0", "also not present")),
        ]
    )
    outcome = FourRoleResearcher(StubRetriever(), llm(transport), CHUNKS.get).run("q")
    assert len(transport.calls) == 3, "bounded: no second retry"
    assert outcome.verdicts[0].quote_found is False
    assert "unverified" in outcome.report.lower()
    assert "Hybrid helps." in outcome.report, "a rejected claim is flagged, not dropped"


def test_retries_can_be_disabled_as_an_arm(db):
    transport = ScriptedTransport(
        [
            {"subquestions": ["Sub one"]},
            claims(("Hybrid helps.", "p1:0", "not present")),
        ]
    )
    outcome = FourRoleResearcher(
        StubRetriever(), llm(transport), CHUNKS.get, max_verify_retries=0
    ).run("q")
    assert len(transport.calls) == 2, "no retry when the arm is off"
    assert outcome.verdicts[0].quote_found is False


def test_more_than_one_retry_is_refused():
    with pytest.raises(ConfigError, match="unbounded"):
        FourRoleResearcher(
            StubRetriever(), llm(ScriptedTransport([])), CHUNKS.get, max_verify_retries=2
        )


def test_an_empty_plan_is_refused(db):
    transport = ScriptedTransport([{"subquestions": []}])
    with pytest.raises(ConfigError, match="no sub-questions"):
        FourRoleResearcher(StubRetriever(), llm(transport), CHUNKS.get).run("q")


def test_the_plan_is_capped_at_the_configured_width(db):
    transport = ScriptedTransport(
        [
            {"subquestions": [f"Sub {i}" for i in range(9)]},
            *[claims((f"C{i}.", "p1:0", "combines BM25 and dense scores")) for i in range(3)],
        ]
    )
    outcome = FourRoleResearcher(
        StubRetriever(), llm(transport), CHUNKS.get, n_subquestions=3
    ).run("q")
    assert len(outcome.subquestions) == 3


def test_both_arms_produce_the_same_outcome_shape(db):
    """So the harness can score them identically -- the point of a baseline."""
    single = SingleAgentResearcher(
        StubRetriever(),
        llm(ScriptedTransport([claims(("A.", "p1:0", "combines BM25 and dense scores"))])),
        CHUNKS.get,
    ).run("q")
    graph = FourRoleResearcher(
        StubRetriever(),
        llm(
            ScriptedTransport(
                [
                    {"subquestions": ["Sub one"]},
                    claims(("A.", "p1:0", "combines BM25 and dense scores")),
                ]
            )
        ),
        CHUNKS.get,
    ).run("q")
    assert set(single.counts) == set(graph.counts)
    assert single.usage.calls >= 1 and graph.usage.calls >= 2


def test_parallelism_is_measurable_not_merely_asserted(db):
    """The graph records both wall-clock and the serial sum of its researchers,
    so "fan-out reduces latency" is a comparison rather than a claim."""
    transport = ScriptedTransport(
        [
            {"subquestions": ["Sub one", "Sub two", "Sub three"]},
            *[claims((f"C{i}.", "p1:0", "combines BM25 and dense scores")) for i in range(3)],
        ]
    )
    outcome = FourRoleResearcher(StubRetriever(), llm(transport), CHUNKS.get).run("q")
    assert "research" in outcome.stage_latency_ms
    assert "research_serial_sum" in outcome.stage_latency_ms
    assert outcome.stage_latency_ms["research_serial_sum"] > 0


def test_single_agent_times_generation_separately_from_retrieval(db):
    transport = ScriptedTransport([claims(("A.", "p1:0", "combines BM25 and dense scores"))])
    outcome = SingleAgentResearcher(StubRetriever(), llm(transport), CHUNKS.get).run("q")
    assert outcome.stage_latency_ms["synthesize"] > 0, "generation must actually be timed"
    assert outcome.stage_latency_ms["retrieve"] > 0
