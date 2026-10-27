"""Tests for claim verification and report rendering.

This is the module that makes "citation-backed" mean something, so the tests
are about the two mechanically different failure classes it must keep apart:

  FABRICATED QUOTE   -- the quote is not in the cited chunk. Decided by
                        str.find after canonicalisation: no model, no variance,
                        no cost. Should be ~0, and any non-zero value is
                        reportable as a hard number.
  UNSUPPORTED CLAIM  -- the quote is real but does not entail the claim.
                        Genuinely fuzzy, needs the judge, and is where the
                        kappa figure matters.

And the rule that keeps the numbers honest: an unverified claim is RENDERED as
unverified, never dropped. Dropping it would hide the failure the evaluation
exists to count and silently inflate groundedness.
"""

from __future__ import annotations

import pytest

from agentforge.agents.verify import (
    CLAIMS_SCHEMA,
    ClaimVerdict,
    render_report,
    verify_claims,
)
from agentforge.llm import _validate

CHUNKS = {
    "p1:0": "Hybrid retrieval combines BM25 and dense scores, evaluated on SciFact.",
    "p2:0": "A cross-encoder rescores candidates jointly with the query.",
    # Ligatures and a line break, as real PDF text has.
    "p3:0": "We evaluate the ﬁrst-stage re­triev-\nal on BEIR’s SciFact split.",
}


def claim(text: str, chunk_id: str, quote: str) -> dict:
    return {"claim": text, "citations": [{"chunk_id": chunk_id, "quote": quote}]}


# ---------------------------------------------------------------- the schema


def test_a_well_formed_claim_set_satisfies_the_schema():
    payload = {"claims": [claim("Hybrid helps.", "p1:0", "combines BM25 and dense scores")]}
    _validate(payload, CLAIMS_SCHEMA)


def test_a_claim_without_citations_is_schema_invalid():
    from agentforge.errors import SchemaViolation

    with pytest.raises(SchemaViolation, match="citations"):
        _validate({"claims": [{"claim": "Unsupported."}]}, CLAIMS_SCHEMA)


def test_a_citation_without_a_quote_is_schema_invalid():
    from agentforge.errors import SchemaViolation

    payload = {"claims": [{"claim": "x", "citations": [{"chunk_id": "p1:0"}]}]}
    with pytest.raises(SchemaViolation, match="quote"):
        _validate(payload, CLAIMS_SCHEMA)


# ------------------------------------------------- deterministic quote check


def test_a_verbatim_quote_is_found():
    verdicts = verify_claims(
        [claim("Hybrid helps.", "p1:0", "combines BM25 and dense scores")], CHUNKS
    )
    assert len(verdicts) == 1
    assert verdicts[0].quote_found is True
    assert verdicts[0].reason is None


def test_a_quote_survives_pdf_artefacts_in_the_source():
    """The model quotes clean text; the chunk has ligatures and a hyphen break."""
    verdicts = verify_claims(
        [claim("It is evaluated on SciFact.", "p3:0", "the first-stage retrieval on BEIR's SciFact split")],
        CHUNKS,
    )
    assert verdicts[0].quote_found is True


def test_a_fabricated_quote_is_detected_without_a_model():
    verdicts = verify_claims(
        [claim("Hybrid helps.", "p1:0", "combines BM25 with learned sparse scores")], CHUNKS
    )
    assert verdicts[0].quote_found is False
    assert verdicts[0].reason == "quote_not_in_chunk"


def test_a_quote_attributed_to_the_wrong_chunk_is_detected():
    verdicts = verify_claims(
        [claim("Reranking helps.", "p1:0", "rescores candidates jointly with the query")],
        CHUNKS,
    )
    assert verdicts[0].quote_found is False


def test_citing_a_chunk_that_was_never_retrieved_is_recorded_not_raised():
    """An invented chunk_id is a model-quality outcome, not a system defect.

    Raising here would mean the tool cannot report its own primary finding.
    """
    verdicts = verify_claims([claim("Something.", "ghost:9", "anything at all")], CHUNKS)
    assert verdicts[0].quote_found is False
    assert verdicts[0].reason == "unknown_chunk"


def test_an_empty_quote_is_a_failure_to_cite():
    verdicts = verify_claims([claim("Something.", "p1:0", "")], CHUNKS)
    assert verdicts[0].quote_found is False
    assert verdicts[0].reason == "empty_quote"


def test_a_claim_with_several_citations_yields_a_verdict_per_citation():
    multi = {
        "claim": "Both stages matter.",
        "citations": [
            {"chunk_id": "p1:0", "quote": "combines BM25 and dense scores"},
            {"chunk_id": "p2:0", "quote": "rescores candidates jointly with the query"},
        ],
    }
    verdicts = verify_claims([multi], CHUNKS)
    assert len(verdicts) == 2
    assert all(v.quote_found for v in verdicts)
    assert {v.claim_idx for v in verdicts} == {0}


def test_claim_indices_are_stable_and_zero_based():
    verdicts = verify_claims(
        [
            claim("One.", "p1:0", "combines BM25 and dense scores"),
            claim("Two.", "p2:0", "rescores candidates jointly"),
        ],
        CHUNKS,
    )
    assert [v.claim_idx for v in verdicts] == [0, 1]


# ------------------------------------------------------------------ rendering


def verdicts_for(entailments: list[str | None]) -> tuple[list[dict], list[ClaimVerdict]]:
    claims = [
        claim(f"Claim {i}.", "p1:0", "combines BM25 and dense scores")
        for i in range(len(entailments))
    ]
    verdicts = [
        ClaimVerdict(
            claim_idx=i,
            claim_text=f"Claim {i}.",
            chunk_id="p1:0",
            quote="combines BM25 and dense scores",
            quote_found=True,
            reason=None,
            entailed=entailment,
        )
        for i, entailment in enumerate(entailments)
    ]
    return claims, verdicts


def test_a_report_renders_entailed_claims_with_footnote_markers():
    claims, verdicts = verdicts_for(["entailed"])
    report = render_report(claims, verdicts, CHUNKS)
    assert "Claim 0." in report
    assert "[^1]" in report
    assert "combines BM25 and dense scores" in report


def test_an_unsupported_claim_is_marked_not_removed():
    """Dropping it would hide the failure the evaluation exists to count."""
    claims, verdicts = verdicts_for(["unsupported"])
    report = render_report(claims, verdicts, CHUNKS)
    assert "Claim 0." in report, "the claim must still appear"
    assert "unverified" in report.lower()


def test_a_fabricated_quote_is_marked_in_the_report():
    claims, _ = verdicts_for([None])
    verdicts = [
        ClaimVerdict(
            claim_idx=0, claim_text="Claim 0.", chunk_id="p1:0", quote="invented text",
            quote_found=False, reason="quote_not_in_chunk", entailed=None,
        )
    ]
    report = render_report(claims, verdicts, CHUNKS)
    assert "Claim 0." in report
    assert "fabricated" in report.lower() or "not found" in report.lower()


def test_a_contradicted_claim_is_marked_distinctly_from_unsupported():
    claims, verdicts = verdicts_for(["contradicted"])
    report = render_report(claims, verdicts, CHUNKS)
    assert "contradicted" in report.lower()


def test_the_report_summarises_verification_counts():
    claims, verdicts = verdicts_for(["entailed", "entailed", "unsupported"])
    report = render_report(claims, verdicts, CHUNKS)
    assert "2" in report and "3" in report  # 2 of 3 verified
