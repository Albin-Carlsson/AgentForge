"""Tests for parser fidelity scoring against ar5iv HTML.

arXiv publishes a LaTeXML rendering of each paper's source, so its heading tags
are independent ground truth for the section boundaries the PDF parser guesses at
from font geometry. Scoring against it turns "heading detection is a heuristic"
into a number, which is the difference between a parser you can describe and a
parser you can defend.

Note what counts as what: a paper with no ar5iv rendering is a gap in COVERAGE,
reported as such, not a parser failure and not a defect.
"""

from __future__ import annotations

import pytest

from agentforge.ingest.fidelity import (
    headings_from_latexml,
    normalise_heading,
    run_in_headings_from_latexml,
    score_sections,
)

LATEXML_HTML = """
<!DOCTYPE html><html><body>
<div class="ltx_page_content">
  <h1 class="ltx_title ltx_title_document">Hybrid Retrieval for Science</h1>
  <div class="ltx_abstract"><h6 class="ltx_title">Abstract</h6><p>We study.</p></div>
  <section class="ltx_section" id="S1">
    <h2 class="ltx_title ltx_title_section">
      <span class="ltx_tag ltx_tag_section">1 </span>Introduction</h2>
    <p>Body text.</p>
    <section class="ltx_subsection" id="S1.SS1">
      <h3 class="ltx_title ltx_title_subsection">
        <span class="ltx_tag ltx_tag_subsection">1.1 </span>Motivation</h3>
    </section>
  </section>
  <section class="ltx_section" id="S2">
    <h2 class="ltx_title ltx_title_section">
      <span class="ltx_tag ltx_tag_section">2 </span>Related Work</h2>
  </section>
  <section class="ltx_bibliography"><h2 class="ltx_title">References</h2></section>
</div></body></html>
"""


def test_headings_are_extracted_from_section_tags():
    headings = headings_from_latexml(LATEXML_HTML)
    assert "1 Introduction" in headings
    assert "1.1 Motivation" in headings
    assert "2 Related Work" in headings


def test_the_document_title_is_not_a_heading():
    assert "Hybrid Retrieval for Science" not in headings_from_latexml(LATEXML_HTML)


def test_nested_tag_spans_are_joined_not_duplicated():
    """ar5iv puts the section number in a nested <span>."""
    headings = headings_from_latexml(LATEXML_HTML)
    assert "Introduction" not in headings  # the bare, un-numbered form
    assert sum(1 for h in headings if "Introduction" in h) == 1


def test_normalisation_strips_numbering_and_case():
    assert normalise_heading("1 Introduction") == "introduction"
    assert normalise_heading("2.1. Related Work") == "related work"
    assert normalise_heading("I. INTRODUCTION") == "introduction"
    assert normalise_heading("A. DENSE AND SPARSE RETRIEVAL") == "dense and sparse retrieval"
    assert normalise_heading("Appendix A Proofs") == "proofs"
    assert normalise_heading("  Related   Work  ") == "related work"


def test_normalisation_makes_differently_numbered_headings_comparable():
    """The PDF and the HTML may number the same section differently."""
    assert normalise_heading("3 Methodology") == normalise_heading("II. METHODOLOGY")


def test_perfect_agreement_scores_one():
    truth = ["1 Introduction", "2 Related Work"]
    found = ["1 Introduction", "2 Related Work"]
    scores = score_sections(found, truth)
    assert scores.recall == 1.0
    assert scores.precision == 1.0
    assert scores.n_truth == 2


def test_a_missed_section_lowers_recall_only():
    truth = ["1 Introduction", "2 Related Work", "3 Method"]
    found = ["1 Introduction", "2 Related Work"]
    scores = score_sections(found, truth)
    assert scores.recall == 2 / 3
    assert scores.precision == 1.0
    assert scores.missed == ["3 Method"]


def test_a_spurious_section_lowers_precision_only():
    truth = ["1 Introduction"]
    found = ["1 Introduction", "0.99 therefore misses about ten thousand"]
    scores = score_sections(found, truth)
    assert scores.recall == 1.0
    assert scores.precision == 0.5
    assert len(scores.spurious) == 1


def test_differently_numbered_but_same_sections_agree():
    scores = score_sections(["I. INTRODUCTION", "II. METHOD"], ["1 Introduction", "2 Method"])
    assert scores.recall == 1.0
    assert scores.precision == 1.0


def test_empty_ground_truth_is_reported_not_scored():
    """No rendering is a coverage gap, not a parser score of zero."""
    scores = score_sections(["1 Introduction"], [])
    assert scores.n_truth == 0
    assert scores.recall is None
    assert scores.precision is None


# --------------------------------------------------------------------------
# Cases from the first real fidelity run, where the SCORER was wrong rather
# than the parser. arXiv:2610.30612 scored recall 0.025 / precision 0.028 --
# near-total disagreement on a paper the parser had actually read correctly --
# because these label forms were not being stripped before comparison.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "heading,expected",
    [
        ("I Introduction", "introduction"),                      # roman, no period
        ("II Related Work", "related work"),
        ("II-A Dense and Sparse Retrieval", "dense and sparse retrieval"),
        ("II-B Hybrid Fusion and the RRF Default", "hybrid fusion and the rrf default"),
        ("A.1. Decomposition of Pointwise Feedback Fitting",
         "decomposition of pointwise feedback fitting"),
        ("A.2. Proof of Theorem 2.1", "proof of theorem 2 1"),
        ("E.1. NDCG@10 on Subsets", "ndcg 10 on subsets"),
        ("3.3 Behavioral diagnostics after the audit",
         "behavioral diagnostics after the audit"),
        ("4.9 Ablation Study", "ablation study"),
    ],
)
def test_real_label_forms_are_stripped(heading, expected):
    assert normalise_heading(heading) == expected


def test_ieee_pdf_headings_match_their_latexml_counterparts():
    """The exact pair that scored 0.025: both describe the same sections."""
    found = ["I. INTRODUCTION", "II. RELATED WORK", "A. DENSE AND SPARSE RETRIEVAL"]
    truth = ["I Introduction", "II Related Work", "II-A Dense and Sparse Retrieval"]
    scores = score_sections(found, truth)
    assert scores.recall == 1.0
    assert scores.precision == 1.0


def test_a_leading_article_is_not_mistaken_for_a_label():
    """'A Model-Agnostic Framework' must keep its leading 'A': a bare capital
    is only a label when a period or compound suffix follows it."""
    assert normalise_heading("A Model-Agnostic Correction Framework") == (
        "a model agnostic correction framework"
    )


@pytest.mark.parametrize(
    "cls,text",
    [
        ("ltx_title ltx_title_abstract", "Abstract"),
        ("ltx_title ltx_title_bibliography", "References"),
        ("ltx_title ltx_title_acknowledgements", "Acknowledgments"),
        ("ltx_title ltx_title_keywords", "Keywords"),
        ("ltx_title ltx_title_appendix", "Appendix A Stress Set"),
    ],
)
def test_front_and_back_matter_count_as_ground_truth_sections(cls, text):
    """These were scored as spurious parser output on every real paper, because
    LaTeXML gives them their own classes. They are real sections."""
    html = f'<html><body><h2 class="{cls}">{text}</h2></body></html>'
    assert headings_from_latexml(html) == [text]


# --------------------------------------------------------------------------
# Text coverage.
#
# Section recall and text coverage measure different failures, and conflating
# them would misread the parser badly. A missed heading does NOT lose text: the
# body beneath it merges into the preceding section, so every character is still
# chunked, embedded and retrievable -- what degrades is the precision of
# `section_path` in a citation. Lost text, by contrast, is unretrievable and is
# the failure that would actually invalidate a retrieval number.
# --------------------------------------------------------------------------

from agentforge.ingest.chunker import Section
from agentforge.ingest.fidelity import text_coverage


def test_full_coverage_when_sections_span_the_document():
    text = "A" * 50 + "B" * 50
    sections = [
        Section(section_path="1", text="A" * 50, page=1, char_offset=0),
        Section(section_path="2", text="B" * 50, page=1, char_offset=50),
    ]
    assert text_coverage(sections, len(text)) == 1.0


def test_coverage_reports_the_fraction_actually_inside_a_section():
    sections = [Section(section_path="1", text="A" * 25, page=1, char_offset=0)]
    assert text_coverage(sections, 100) == 0.25


def test_front_matter_dropped_before_the_first_heading_shows_as_lost_coverage():
    """Authors, affiliations and the title sit before the first heading and are
    deliberately not part of any section -- so coverage is below 1.0 even on a
    perfect parse, and the metric is read as a floor rather than a target."""
    sections = [Section(section_path="1 Introduction", text="B" * 80, page=1, char_offset=20)]
    assert text_coverage(sections, 100) == 0.8


def test_coverage_of_an_empty_document_is_refused():
    with pytest.raises(ValueError, match="empty"):
        text_coverage([], 0)


def test_run_in_paragraph_headings_are_excluded_from_block_level_truth():
    """\\paragraph{} renders inline with its prose, so a line-based PDF parser
    cannot detect it. On arXiv:2610.32318 these are 25 of 38 ground-truth
    headings and alone pushed apparent recall to 0.158."""
    html = (
        '<html><body>'
        '<h2 class="ltx_title ltx_title_section">1 Introduction</h2>'
        '<h6 class="ltx_title ltx_title_paragraph">LLM-based rerankers.</h6>'
        '<h6 class="ltx_title ltx_title_paragraph">Terms.</h6>'
        '</body></html>'
    )
    assert headings_from_latexml(html) == ["1 Introduction"]
    assert run_in_headings_from_latexml(html) == ["LLM-based rerankers.", "Terms."]
