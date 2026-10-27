"""The two research arms: a single-agent baseline and a four-role graph.

The baseline is not a straw man. It is the measurement that justifies the word
"multi-agent" existing on a resume at all: without it, "why more than one
agent?" is a question of taste, and with it the answer is a table.

The four-role graph is `Planner -> parallel Researchers -> Verifier ->
Synthesizer`, justified on three grounds, two of which are about correctness
rather than organisation:

1. **Context isolation.** A single agent accumulates every retrieved chunk from
   every sub-question into one window; by sub-question five, attention is
   diluted by mostly-irrelevant text. Each Researcher here holds only its own
   sub-question's evidence -- enforced by construction, and by a test.
2. **Adversarial separation.** The Verifier sees a claim and the span it cites,
   and nothing else: not the question, not the other claims, not the
   generator's reasoning. A verifier sharing context with the writer
   rubber-stamps its own prior reasoning, which is what turns evidence
   verification from a real check into decoration.
3. **Parallelism.** Sub-questions are independent, so wall-clock latency scales
   with the slowest rather than the sum.

The verify edge is bounded at ONE retry. Unbounded self-correction is how these
graphs become unpredictable in cost and unmeasurable in latency; and whether the
retry rescues claims or merely burns tokens is itself an arm the harness can
evaluate.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from agentforge.agents import prompts
from agentforge.agents.verify import (
    CLAIMS_SCHEMA,
    ENTAILMENT_SCHEMA,
    SUBQUESTIONS_SCHEMA,
    ClaimVerdict,
    counts,
    render_report,
    verify_claims,
)
from agentforge.errors import ConfigError
from agentforge.llm import LLMClient, Usage

ChunkLookup = Callable[[str], str | None]

MAX_PARALLEL_RESEARCHERS = 4


@dataclass(frozen=True)
class _ResearchStep:
    """One Researcher's output, with its stages timed separately."""

    claims: list[dict[str, Any]]
    texts: dict[str, str]
    chunk_ids: list[str]
    retrieve_ms: float
    generate_ms: float


@dataclass
class ResearchOutcome:
    question: str
    subquestions: list[str]
    claims: list[dict[str, Any]]
    verdicts: list[ClaimVerdict]
    report: str
    stage_latency_ms: dict[str, float]
    usage: Usage
    counts: dict[str, int | float] = field(default_factory=dict)
    retrieved: dict[str, list[str]] = field(default_factory=dict)


def format_context(chunk_ids: list[str], lookup: ChunkLookup) -> tuple[str, dict[str, str]]:
    """Render retrieved chunks as numbered, explicitly-identified sources.

    The chunk_id is printed in the header because the model must cite it back
    verbatim; a positional reference ("source 3") would be unresolvable once the
    ranking changed.
    """
    blocks: list[str] = []
    texts: dict[str, str] = {}
    for chunk_id in chunk_ids:
        text = lookup(chunk_id)
        if text is None:
            continue
        texts[chunk_id] = text
        blocks.append(f"[chunk_id: {chunk_id}]\n{text}")
    return "\n\n".join(blocks), texts


class _Base:
    def __init__(self, retriever: Any, llm: LLMClient, lookup: ChunkLookup) -> None:
        self.retriever = retriever
        self.llm = llm
        self.lookup = lookup

    def _research(self, subquestion: str) -> _ResearchStep:
        """Retrieve for one sub-question and extract claims from only its evidence."""
        started = time.perf_counter()
        trace = self.retriever.retrieve(subquestion)
        retrieve_ms = (time.perf_counter() - started) * 1000

        chunk_ids = [hit.chunk_id for hit in trace.final]
        context, texts = format_context(chunk_ids, self.lookup)

        started = time.perf_counter()
        response = self.llm.complete(
            prompts.RESEARCHER_SYSTEM,
            f"Sub-question: {subquestion}\n\nSources:\n\n{context}",
            CLAIMS_SCHEMA,
        )
        generate_ms = (time.perf_counter() - started) * 1000

        return _ResearchStep(
            claims=response.parsed["claims"],
            texts=texts,
            chunk_ids=chunk_ids,
            retrieve_ms=retrieve_ms,
            generate_ms=generate_ms,
        )

    def _judge(self, verdicts: list[ClaimVerdict]) -> list[ClaimVerdict]:
        """Entailment, on claims whose quote was found. One call per citation.

        The prompt carries a claim and a quote and nothing else -- the whole
        point of putting this behind its own call.
        """
        judged: list[ClaimVerdict] = []
        for verdict in verdicts:
            if not verdict.quote_found:
                judged.append(verdict)
                continue
            response = self.llm.complete(
                prompts.JUDGE_SYSTEM,
                f"Claim:\n{verdict.claim_text}\n\nQuote:\n{verdict.quote}",
                ENTAILMENT_SCHEMA,
            )
            judged.append(
                ClaimVerdict(
                    claim_idx=verdict.claim_idx,
                    claim_text=verdict.claim_text,
                    chunk_id=verdict.chunk_id,
                    quote=verdict.quote,
                    quote_found=True,
                    reason=None,
                    entailed=response.parsed["verdict"],
                    verify_pass=verdict.verify_pass,
                )
            )
        return judged


class SingleAgentResearcher(_Base):
    """The baseline arm: one retrieval, one generation, same verification.

    Everything downstream of generation is identical to the graph's, so a
    comparison between the two isolates the orchestration rather than the
    scoring.
    """

    def run(self, question: str) -> ResearchOutcome:
        step = self._research(question)
        started = time.perf_counter()
        verdicts = verify_claims(step.claims, step.texts)
        verify_ms = (time.perf_counter() - started) * 1000

        return ResearchOutcome(
            question=question,
            subquestions=[question],
            claims=step.claims,
            verdicts=verdicts,
            report=render_report(step.claims, verdicts, step.texts),
            stage_latency_ms={
                "retrieve": step.retrieve_ms,
                "synthesize": step.generate_ms,
                "verify": verify_ms,
            },
            usage=self.llm.usage,
            counts=counts(verdicts),
            retrieved={question: step.chunk_ids},
        )


class FourRoleResearcher(_Base):
    def __init__(
        self,
        retriever: Any,
        llm: LLMClient,
        lookup: ChunkLookup,
        n_subquestions: int = 4,
        max_verify_retries: int = 1,
        judge_enabled: bool = False,
    ) -> None:
        super().__init__(retriever, llm, lookup)
        if max_verify_retries > 1:
            raise ConfigError(
                f"max_verify_retries={max_verify_retries}: unbounded self-correction "
                "makes the graph's cost unpredictable and its latency distribution "
                "unmeasurable"
            )
        self.n_subquestions = n_subquestions
        self.max_verify_retries = max_verify_retries
        self.judge_enabled = judge_enabled

    def _plan(self, question: str) -> list[str]:
        response = self.llm.complete(
            prompts.PLANNER_SYSTEM.format(max_subquestions=self.n_subquestions),
            f"Research question: {question}",
            SUBQUESTIONS_SCHEMA,
        )
        subquestions = [s.strip() for s in response.parsed["subquestions"] if s.strip()]
        if not subquestions:
            raise ConfigError(
                "the Planner returned no sub-questions; refusing to continue with an "
                "empty plan rather than silently falling back to the raw question"
            )
        return subquestions[: self.n_subquestions]

    def run(self, question: str) -> ResearchOutcome:
        stage_ms: dict[str, float] = {}

        started = time.perf_counter()
        subquestions = self._plan(question)
        stage_ms["plan"] = (time.perf_counter() - started) * 1000

        # Researchers run concurrently: the sub-questions are independent by
        # construction, so latency scales with the slowest rather than the sum.
        started = time.perf_counter()
        if len(subquestions) == 1:
            results = [self._research(subquestions[0])]
        else:
            with ThreadPoolExecutor(
                max_workers=min(MAX_PARALLEL_RESEARCHERS, len(subquestions))
            ) as pool:
                results = list(pool.map(self._research, subquestions))
        stage_ms["research"] = (time.perf_counter() - started) * 1000

        all_claims: list[dict[str, Any]] = []
        texts: dict[str, str] = {}
        retrieved: dict[str, list[str]] = {}
        for subquestion, step in zip(subquestions, results, strict=True):
            all_claims.extend(step.claims)
            texts.update(step.texts)
            retrieved[subquestion] = step.chunk_ids

        # Researchers ran concurrently, so the wall-clock figure above is the
        # useful one; the per-researcher sum is recorded separately to make the
        # parallelism claim checkable rather than asserted.
        stage_ms["research_serial_sum"] = sum(
            step.retrieve_ms + step.generate_ms for step in results
        )

        started = time.perf_counter()
        verdicts = verify_claims(all_claims, texts)
        stage_ms["verify"] = (time.perf_counter() - started) * 1000

        # Bounded retry: one pass, with the rejection fed back to a Researcher.
        if self.max_verify_retries and any(not v.quote_found for v in verdicts):
            started = time.perf_counter()
            all_claims, verdicts = self._retry_rejected(
                subquestions, all_claims, verdicts, texts
            )
            stage_ms["retry"] = (time.perf_counter() - started) * 1000

        if self.judge_enabled:
            started = time.perf_counter()
            verdicts = self._judge(verdicts)
            stage_ms["judge"] = (time.perf_counter() - started) * 1000

        return ResearchOutcome(
            question=question,
            subquestions=subquestions,
            claims=all_claims,
            verdicts=verdicts,
            report=render_report(all_claims, verdicts, texts),
            stage_latency_ms=stage_ms,
            usage=self.llm.usage,
            counts=counts(verdicts),
            retrieved=retrieved,
        )

    def _retry_rejected(
        self,
        subquestions: list[str],
        claims: list[dict[str, Any]],
        verdicts: list[ClaimVerdict],
        texts: dict[str, str],
    ) -> tuple[list[dict[str, Any]], list[ClaimVerdict]]:
        """One re-research pass for the rejected claims.

        The rejected claim and the reason go back to a Researcher over the same
        evidence. A claim still rejected after this ships FLAGGED: dropping it
        would hide the failure the evaluation exists to count.
        """
        rejected_idx = sorted({v.claim_idx for v in verdicts if not v.quote_found})
        context, _ = format_context(list(texts), self.lookup)
        rejected_text = "\n".join(f"- {claims[i]['claim']}" for i in rejected_idx)

        response = self.llm.complete(
            prompts.RESEARCHER_SYSTEM,
            f"Sub-question: {subquestions[0]}\n\n"
            "The following claims were rejected because their quotes could not be "
            "found verbatim in the cited source. Re-state only the ones the sources "
            "genuinely support, copying each quote exactly. Omit any the sources do "
            f"not support.\n\n{rejected_text}\n\nSources:\n\n{context}",
            CLAIMS_SCHEMA,
        )

        replacements = response.parsed["claims"]
        kept = [claims[i] for i in range(len(claims)) if i not in set(rejected_idx)]
        merged = kept + replacements
        new_verdicts = verify_claims(merged, texts)
        # Mark everything from this pass, so the harness can measure whether the
        # retry rescued claims or merely spent tokens.
        return merged, [
            ClaimVerdict(
                claim_idx=v.claim_idx,
                claim_text=v.claim_text,
                chunk_id=v.chunk_id,
                quote=v.quote,
                quote_found=v.quote_found,
                reason=v.reason,
                entailed=v.entailed,
                verify_pass=1,
            )
            for v in new_verdicts
        ]
