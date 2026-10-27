"""Structured claims, deterministic quote verification, and report rendering.

The Synthesizer emits claims as STRUCTURE, not prose, and readable output is
rendered from them. Two reasons, both of which protect the measurement:

**Error stays out of the metric path.** Decomposing prose into claims after the
fact puts an extra LLM call inside the measurement: the decomposer's own error
rate -- mis-split compound claims, dropped hedges, citations attached to the
wrong half of a sentence -- is baked into every hallucination number, and the
judge's kappa then partly measures the decomposer. Emitting structure removes
that stage entirely.

**Fabrication detection becomes deterministic.** A verbatim quote either is or
is not a substring of its cited chunk. That is `str.find` after canonical
normalisation: no model call, no variance, no cost. It splits the hallucination
metric into two mechanically different classes --

    fabricated quote   deterministic, should be ~0, any non-zero is a hard fact
    unsupported claim  the quote is real but does not entail the claim; fuzzy,
                       needs the judge, and is where kappa matters

Claims that fail either check are RENDERED AS UNVERIFIED, never dropped.
Dropping them would hide the failure this project exists to count and would
silently inflate its own groundedness numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agentforge.ingest.normalize import quote_found

# Enforced API-side via output_config.format, and re-validated locally.
CLAIMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "chunk_id": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["chunk_id", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["claim", "citations"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

SUBQUESTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "subquestions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["subquestions"],
    "additionalProperties": False,
}

ENTAILMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["entailed", "contradicted", "unsupported"],
        },
    },
    "required": ["verdict"],
    "additionalProperties": False,
}

# Why a claim could not be verified. Each value is a count in the report.
REASON_UNKNOWN_CHUNK = "unknown_chunk"
REASON_EMPTY_QUOTE = "empty_quote"
REASON_QUOTE_NOT_IN_CHUNK = "quote_not_in_chunk"


@dataclass(frozen=True)
class ClaimVerdict:
    claim_idx: int
    claim_text: str
    chunk_id: str
    quote: str
    quote_found: bool
    reason: str | None
    entailed: str | None = None
    verify_pass: int = 0

    @property
    def verified(self) -> bool:
        return self.quote_found and self.entailed == "entailed"


def verify_claims(
    claims: list[dict[str, Any]], chunk_texts: dict[str, str]
) -> list[ClaimVerdict]:
    """One verdict per citation. No model calls; no exceptions on bad output.

    A citation naming a chunk that was never retrieved is a MODEL-QUALITY
    OUTCOME, recorded as such -- not a system defect. Raising would mean the
    tool cannot report its own primary finding.
    """
    verdicts: list[ClaimVerdict] = []
    for index, claim in enumerate(claims):
        citations = claim.get("citations") or []
        if not citations:
            verdicts.append(
                ClaimVerdict(
                    claim_idx=index,
                    claim_text=claim["claim"],
                    chunk_id="",
                    quote="",
                    quote_found=False,
                    reason=REASON_EMPTY_QUOTE,
                )
            )
            continue

        for citation in citations:
            chunk_id = citation["chunk_id"]
            quote = citation["quote"]
            if chunk_id not in chunk_texts:
                found, reason = False, REASON_UNKNOWN_CHUNK
            elif not quote.strip():
                found, reason = False, REASON_EMPTY_QUOTE
            elif quote_found(quote, chunk_texts[chunk_id]):
                found, reason = True, None
            else:
                found, reason = False, REASON_QUOTE_NOT_IN_CHUNK

            verdicts.append(
                ClaimVerdict(
                    claim_idx=index,
                    claim_text=claim["claim"],
                    chunk_id=chunk_id,
                    quote=quote,
                    quote_found=found,
                    reason=reason,
                )
            )
    return verdicts


def counts(verdicts: list[ClaimVerdict]) -> dict[str, int | float]:
    """The numbers the evaluation reports, computed from verdicts alone."""
    total = len(verdicts)
    fabricated = sum(1 for v in verdicts if v.reason == REASON_QUOTE_NOT_IN_CHUNK)
    unknown = sum(1 for v in verdicts if v.reason == REASON_UNKNOWN_CHUNK)
    empty = sum(1 for v in verdicts if v.reason == REASON_EMPTY_QUOTE)
    entailed = sum(1 for v in verdicts if v.verified)
    contradicted = sum(1 for v in verdicts if v.entailed == "contradicted")
    unsupported = sum(1 for v in verdicts if v.quote_found and v.entailed == "unsupported")
    return {
        "citations": total,
        "verified": entailed,
        "fabricated_quotes": fabricated,
        "unknown_chunks": unknown,
        "empty_quotes": empty,
        "contradicted": contradicted,
        "unsupported": unsupported,
        "groundedness": entailed / total if total else 0.0,
        # Deterministic half: no judge involved, so this number has no variance.
        "fabrication_rate": (fabricated + unknown) / total if total else 0.0,
    }


_MARKS = {
    REASON_QUOTE_NOT_IN_CHUNK: "unverified: quote not found in the cited source (fabricated quote)",
    REASON_UNKNOWN_CHUNK: "unverified: cites a source that was not retrieved",
    REASON_EMPTY_QUOTE: "unverified: no quote supplied",
}


def render_report(
    claims: list[dict[str, Any]],
    verdicts: list[ClaimVerdict],
    chunk_texts: dict[str, str],
    sources: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Render claims as a cited markdown report.

    Unverified claims appear with an explicit marker. A reader can therefore
    tell the difference between what the system supported and what it merely
    asserted -- which is the whole point of the exercise.
    """
    by_claim: dict[int, list[ClaimVerdict]] = {}
    for verdict in verdicts:
        by_claim.setdefault(verdict.claim_idx, []).append(verdict)

    lines: list[str] = []
    footnotes: list[str] = []
    marker = 0

    for index, claim in enumerate(claims):
        claim_verdicts = by_claim.get(index, [])
        markers: list[str] = []
        notes: list[str] = []

        for verdict in claim_verdicts:
            if verdict.quote_found:
                marker += 1
                markers.append(f"[^{marker}]")
                source = (sources or {}).get(verdict.chunk_id, {})
                where = ""
                if source.get("page") is not None:
                    where = f", p. {source['page']}"
                title = source.get("title") or verdict.chunk_id
                footnotes.append(
                    f'[^{marker}]: {title}{where} — "{verdict.quote}" '
                    f"(`{verdict.chunk_id}`)"
                )
                if verdict.entailed == "contradicted":
                    notes.append("**contradicted by its own cited source**")
                elif verdict.entailed == "unsupported":
                    notes.append("**unverified: the cited quote does not support this claim**")
            else:
                notes.append(f"**{_MARKS[verdict.reason or REASON_EMPTY_QUOTE]}**")

        suffix = f" {' '.join(markers)}" if markers else ""
        lines.append(f"{claim['claim']}{suffix}")
        for note in notes:
            lines.append(f"  - {note}")
        lines.append("")

    stats = counts(verdicts)
    body = "\n".join(lines).rstrip()
    summary = (
        f"**Verification:** {stats['verified']} of {stats['citations']} citations "
        f"verified against their sources"
    )
    details = []
    if stats["fabricated_quotes"]:
        details.append(f"{stats['fabricated_quotes']} fabricated quote(s)")
    if stats["unknown_chunks"]:
        details.append(f"{stats['unknown_chunks']} citation(s) to unretrieved sources")
    if stats["contradicted"]:
        details.append(f"{stats['contradicted']} contradicted")
    if stats["unsupported"]:
        details.append(f"{stats['unsupported']} unsupported")
    if details:
        summary += " — " + ", ".join(details)

    parts = [body, "", "---", "", summary]
    if footnotes:
        parts += ["", *footnotes]
    return "\n".join(parts) + "\n"
