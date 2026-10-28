"""Tests for the judge and its validation.

An unvalidated LLM judge makes every groundedness number above it decorative,
so the machinery that licenses it is tested as carefully as the thing it
measures:

  * kappa is computed against reference labels, and a judge below the floor is
    REFUSED rather than used with a caveat;
  * kappa against labels that are still model-generated is reported as such,
    because calling that "human agreement" is the overstatement this project
    exists to avoid;
  * the judge sees a claim and a quote only -- no question, no sibling claims.
"""

from __future__ import annotations

import pytest

from agentforge.errors import MeasurementError
from agentforge.eval.judge import (
    EntailmentJudge,
    JudgeValidation,
    store_reference_pairs,
    validate_judge,
)
from agentforge.llm import LLMClient

PAIRS = [
    ("Hybrid retrieval combines two retrievers.", "combines BM25 and dense scores", "entailed"),
    ("Reranking always helps.", "reranking degraded nDCG on SciFact", "contradicted"),
    ("The corpus has 5183 documents.", "we evaluate on BEIR", "unsupported"),
]


class Replay:
    """Answers by matching the CLAIM in the prompt, not by call order.

    Reference pairs come back ordered by pair_id (a content hash), so an
    order-dependent fake pairs verdicts with the wrong claims -- which is
    exactly how this test first reported kappa = -0.5 for a judge that agreed
    perfectly. Matching on content also makes the test check that each claim
    reaches the judge with its own quote.
    """

    def __init__(self, by_claim: dict[str, str] | str) -> None:
        self.by_claim = by_claim
        self.prompts: list[str] = []

    def __call__(self, *, model, system, messages, max_tokens, schema, thinking):
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        if isinstance(self.by_claim, str):
            verdict = self.by_claim
        else:
            matches = [v for claim, v in self.by_claim.items() if claim in prompt]
            if len(matches) != 1:
                raise AssertionError(f"no unique scripted verdict for prompt: {prompt!r}")
            verdict = matches[0]
        return {"parsed": {"verdict": verdict}, "input_tokens": 50, "output_tokens": 5}


def judge(by_claim: dict[str, str] | str, model: str = "claude-haiku-4-5") -> EntailmentJudge:
    return EntailmentJudge(
        LLMClient(model=model, transport=Replay(by_claim), cache_bypass=True),
        prompt_version="v1",
    )


def agreeing() -> dict[str, str]:
    return {claim: label for claim, _, label in PAIRS}


def disagreeing() -> dict[str, str]:
    """Each pair gets a label other than its reference one."""
    rotate = {"entailed": "contradicted", "contradicted": "unsupported",
              "unsupported": "entailed"}
    return {claim: rotate[label] for claim, _, label in PAIRS}


def test_the_judge_sees_only_a_claim_and_a_quote():
    j = judge("entailed")
    j.judge("Hybrid helps.", "combines BM25 and dense scores")
    prompt = j.llm.transport.prompts[0]  # type: ignore[union-attr]
    assert "Hybrid helps." in prompt
    assert "combines BM25 and dense scores" in prompt
    assert len(prompt) < 400, "no question, no sibling claims, no reasoning"


def test_the_judge_returns_one_of_three_verdicts():
    assert judge("entailed").judge("c", "q") == "entailed"
    assert judge("contradicted").judge("c", "q") == "contradicted"
    assert judge("unsupported").judge("c", "q") == "unsupported"


def test_perfect_agreement_yields_kappa_one(db):
    store_reference_pairs(PAIRS, label_source="human-reviewed")
    result = validate_judge(judge(agreeing()), min_kappa=0.7)
    assert isinstance(result, JudgeValidation)
    assert result.kappa == pytest.approx(1.0)
    assert result.agreement == pytest.approx(1.0)
    assert result.n_pairs == 3
    assert result.usable is True


def test_a_judge_that_disagrees_is_refused(db):
    store_reference_pairs(PAIRS, label_source="human-reviewed")
    # Shifted labels: agreement no better than chance.
    result = validate_judge(judge(disagreeing()), min_kappa=0.7)
    assert result.kappa < 0.7
    assert result.usable is False
    assert "below" in result.note.lower()


def test_a_constant_judge_is_caught_despite_high_agreement(db):
    """A judge answering 'entailed' always scores well on raw agreement in a
    skewed set. Kappa is the number that catches it."""
    skewed = [(f"c{i}", f"q{i}", "entailed") for i in range(9)]
    skewed.append(("c9", "q9", "unsupported"))
    store_reference_pairs(skewed, label_source="human-reviewed")
    result = validate_judge(judge("entailed"), min_kappa=0.7)
    assert result.agreement == pytest.approx(0.9)
    assert result.kappa < 0.7, "high agreement, no skill"
    assert result.usable is False


def test_kappa_against_unreviewed_labels_is_labelled_as_such(db):
    """Calling model-generated labels 'human agreement' is the overstatement
    this whole project is organised to avoid."""
    store_reference_pairs(PAIRS, label_source="model-pending-review")
    result = validate_judge(judge(agreeing()), min_kappa=0.7)
    assert result.label_source == "model-pending-review"
    assert "pending human review" in result.note.lower()
    assert result.human_validated is False


def test_reviewed_labels_are_reported_as_human_validated(db):
    store_reference_pairs(PAIRS, label_source="human-reviewed")
    result = validate_judge(judge(agreeing()), min_kappa=0.7)
    assert result.human_validated is True


def test_validation_without_reference_pairs_is_refused(db):
    with pytest.raises(MeasurementError, match="no reference pairs"):
        validate_judge(judge("entailed"), min_kappa=0.7)


def test_validation_is_persisted_for_the_report(db):
    from agentforge.db.conn import pool

    store_reference_pairs(PAIRS, label_source="human-reviewed")
    validate_judge(judge(agreeing()), min_kappa=0.7)
    with pool().connection() as conn:
        row = conn.execute(
            "SELECT kappa, n_pairs, label_source FROM judge_validation "
            "WHERE judge_model = %s",
            ("claude-haiku-4-5",),
        ).fetchone()
    assert row is not None
    assert row["n_pairs"] == 3
    assert row["label_source"] == "human-reviewed"


def test_storing_pairs_twice_does_not_duplicate_them(db):
    store_reference_pairs(PAIRS, label_source="human-reviewed")
    store_reference_pairs(PAIRS, label_source="human-reviewed")
    result = validate_judge(judge(agreeing()), min_kappa=0.7)
    assert result.n_pairs == 3


def test_reference_file_rejects_a_quote_not_in_its_chunk(db, probe_corpus, tmp_path):
    import json

    from agentforge.eval.judge import load_reference_file

    path = tmp_path / "ref.jsonl"
    path.write_text(json.dumps({"chunk_id": "p1:0", "quote": "words that are not there",
                                "claim": "x", "label": "entailed"}) + "\n")
    with pytest.raises(MeasurementError, match="not verbatim"):
        load_reference_file(str(path))


def test_reference_reload_never_overwrites_a_human_review(db, probe_corpus, tmp_path):
    import json

    from agentforge.eval.judge import (
        HUMAN_REVIEWED,
        load_reference_file,
        mark_reviewed,
        reference_pairs,
    )

    row = {"chunk_id": "p1:0", "quote": "combines BM25 and dense scores",
           "claim": "Hybrid retrieval combines BM25 and dense scores.", "label": "entailed"}
    path = tmp_path / "ref.jsonl"
    path.write_text(json.dumps(row) + "\n")
    load_reference_file(str(path))
    pair = reference_pairs()[0]
    mark_reviewed(pair["pair_id"], "unsupported")
    load_reference_file(str(path))
    after = reference_pairs()[0]
    assert (after["label"], after["label_source"]) == ("unsupported", HUMAN_REVIEWED)
