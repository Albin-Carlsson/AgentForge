"""Tests for PDF parsing.

Built against synthetic PDFs generated in-process rather than downloaded
fixtures, so heading detection and offset mapping are deterministic. A real
arXiv paper is exercised separately by the parser-fidelity check against ar5iv
HTML, which is a measurement rather than a test.

The load-bearing property here is the span map: a chunk knows its character
range, and `bbox_for` turns that range into page rectangles. That is what makes
a citation a highlightable region on a page instead of a filename.
"""

from __future__ import annotations

import pathlib

import pymupdf
import pytest

from agentforge.errors import IngestError
from agentforge.ingest.normalize import canonical
from agentforge.ingest.pdf import bbox_for, parse_pdf


def write_pdf(path: pathlib.Path, pages: list[list[tuple[str, float]]]) -> pathlib.Path:
    """Each page is a list of (text, fontsize); larger sizes read as headings."""
    doc = pymupdf.open()
    for blocks in pages:
        page = doc.new_page(width=612, height=792)
        y = 72.0
        for text, size in blocks:
            page.insert_textbox(
                pymupdf.Rect(72, y, 540, y + 40 + 14 * (len(text) // 60)),
                text,
                fontsize=size,
                fontname="helv",
            )
            y += 48 + 14 * (len(text) // 60)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def paper(tmp_path: pathlib.Path) -> pathlib.Path:
    body = (
        "We study hybrid retrieval over scientific abstracts and report nDCG@10 "
        "across several configurations of the first-stage retriever, with the "
        "text deliberately long enough to wrap across several rendered lines."
    )
    method = (
        "The reranker rescores the top candidates jointly with the query. "
        "We evaluate on BEIR and report paired confidence intervals."
    )
    return write_pdf(
        tmp_path / "paper.pdf",
        [
            [("Hybrid Retrieval for Science", 18.0), ("1 Introduction", 13.0), (body, 10.0)],
            [("2 Method", 13.0), (method, 10.0)],
        ],
    )


def test_sections_are_detected_in_order(paper):
    parsed = parse_pdf(paper)
    assert [s.section_path for s in parsed.sections] == ["1 Introduction", "2 Method"]


def test_section_pages_are_one_indexed(paper):
    parsed = parse_pdf(paper)
    pages = {s.section_path: s.page for s in parsed.sections}
    assert pages["1 Introduction"] == 1
    assert pages["2 Method"] == 2


def test_title_is_the_largest_text_on_the_first_page(paper):
    assert parse_pdf(paper).title == "Hybrid Retrieval for Science"


def test_section_offsets_resolve_into_the_document_text(paper):
    parsed = parse_pdf(paper)
    for section in parsed.sections:
        start = section.char_offset
        assert parsed.text[start : start + len(section.text)] == section.text


def test_each_extracted_span_is_in_canonical_form(paper):
    """Every block is canonicalised on extraction.

    Ligature folding itself is covered in test_normalize.py with real
    codepoints: a synthetic PDF cannot exercise it, because PyMuPDF's built-in
    helv font cannot encode U+FB01 and friends -- they extract as "?". What is
    testable here, and what matters, is that the parser puts each block through
    `canonical`, so line-wrapped text arrives with its whitespace reflowed.
    """
    parsed = parse_pdf(paper)
    for span in parsed.spans:
        piece = parsed.text[span.char_start : span.char_end]
        assert canonical(piece) == piece, repr(piece)
        assert "\n" not in piece
        assert "  " not in piece


def test_line_wrapped_text_is_reflowed_into_one_span(paper):
    """The body wraps over several PDF lines but must read as continuous prose."""
    parsed = parse_pdf(paper)
    assert "first-stage retriever" in parsed.text


def test_spans_cover_every_character_of_the_document(paper):
    parsed = parse_pdf(paper)
    covered = sum(span.char_end - span.char_start for span in parsed.spans)
    assert covered > 0
    assert covered <= len(parsed.text)
    for span in parsed.spans:
        assert parsed.text[span.char_start : span.char_end].strip()


def test_bbox_lookup_returns_the_page_and_rectangle_for_a_range(paper):
    parsed = parse_pdf(paper)
    section = next(s for s in parsed.sections if s.section_path == "2 Method")
    located = bbox_for(parsed.spans, section.char_offset, section.char_offset + 40)
    assert located, "a citation range must resolve to at least one page rectangle"
    page, bbox = located[0]
    assert page == 2
    assert len(bbox) == 4
    x0, y0, x1, y1 = bbox
    assert x1 > x0 and y1 > y0


def test_bbox_lookup_outside_the_document_returns_nothing(paper):
    parsed = parse_pdf(paper)
    assert bbox_for(parsed.spans, 10**9, 10**9 + 10) == []


def test_a_pdf_with_no_headings_raises(tmp_path):
    """For a paper corpus, no detectable sections means the parser failed."""
    flat = write_pdf(tmp_path / "flat.pdf", [[("just body text with no headings at all", 10.0)]])
    with pytest.raises(IngestError, match="no sections"):
        parse_pdf(flat)


def test_a_missing_file_raises(tmp_path):
    with pytest.raises(IngestError, match="cannot open"):
        parse_pdf(tmp_path / "absent.pdf")


def test_a_corrupt_file_raises(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7\nnot actually a pdf\n")
    with pytest.raises(IngestError, match="cannot open"):
        parse_pdf(broken)


def test_page_count_and_parser_version_are_reported(paper):
    """Both are recorded on the document row: the parser is part of the result."""
    parsed = parse_pdf(paper)
    assert parsed.n_pages == 2
    assert parsed.parser_version.startswith("pymupdf-")


def test_headings_merged_into_one_block_with_their_body_are_still_detected(tmp_path):
    """The failure this test was written for, from a real paper.

    PyMuPDF returns one text block per layout region, and in ACM-style
    two-column papers a section heading and the paragraph beneath it land in the
    SAME block. Detecting headings per block therefore saw a 40-word blob and
    rejected it, and arXiv:2610.30612 parsed as only ['ACKNOWLEDGMENT',
    'REFERENCES'] -- its entire body silently unretrievable. Headings are
    detected per line.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    # One textbox: heading line, then body lines. One block, several lines.
    page.insert_textbox(
        pymupdf.Rect(72, 72, 540, 300),
        "1 INTRODUCTION\n"
        "Retrieval augmented generation combines a retriever with a generator. "
        "We study the fusion of lexical and dense scores across several corpora "
        "and report paired confidence intervals throughout this paper.",
        fontsize=10,
        fontname="helv",
    )
    page.insert_textbox(
        pymupdf.Rect(72, 320, 540, 500),
        "2 RELATED WORK\n"
        "Prior work on reciprocal rank fusion established the rank-based "
        "formulation we adopt here for combining retrievers.",
        fontsize=10,
        fontname="helv",
    )
    path = tmp_path / "acm.pdf"
    doc.save(path)
    doc.close()

    parsed = parse_pdf(path)
    assert [s.section_path for s in parsed.sections] == ["1 INTRODUCTION", "2 RELATED WORK"]
    intro = parsed.sections[0]
    assert "Retrieval augmented generation" in intro.text
    assert "1 INTRODUCTION" not in intro.text, "the heading is not part of its own body"
    assert parsed.text[intro.char_offset : intro.char_offset + len(intro.text)] == intro.text


def test_a_heading_line_does_not_absorb_the_following_body_line(tmp_path):
    """Same-size headings must not swallow their paragraph into section_path."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(
        pymupdf.Rect(72, 72, 540, 400),
        "3 METHOD\nWe describe the approach.\n4 RESULTS\nWe report the outcome.",
        fontsize=10,
        fontname="helv",
    )
    path = tmp_path / "tight.pdf"
    doc.save(path)
    doc.close()

    parsed = parse_pdf(path)
    assert [s.section_path for s in parsed.sections] == ["3 METHOD", "4 RESULTS"]
    assert parsed.sections[0].text.strip() == "We describe the approach."


# --------------------------------------------------------------------------
# Heading-detection predicates.
#
# These are unit tests rather than synthetic-PDF tests on purpose. Each case
# below is a line taken verbatim from a real arXiv paper that the parser got
# wrong, with its measured font size and direction. Synthetic PDFs could not
# reproduce any of them -- PyMuPDF's own text insertion does not produce the
# same line geometry a LaTeX-rendered paper does -- so testing through a
# generated PDF would have passed while the real bug stayed live.
# --------------------------------------------------------------------------

from agentforge.ingest.pdf import _is_heading, _is_horizontal, _Line


def line(text: str, size: float = 9.96, page: int = 1, bold: bool = False) -> _Line:
    return _Line(
        text=text, size=size, page=page, bold=bold, bbox=(72.0, 100.0, 540.0, 112.0)
    )


BODY = 9.96


def test_horizontal_text_is_content_and_rotated_text_is_not():
    """arXiv:2610.29464 stamps its left margin at size 20.0, dir=(0,-1) --
    LARGER than the paper's real 17.22pt title, so size alone picked the stamp.
    """
    assert _is_horizontal((1.0, 0.0))
    assert not _is_horizontal((0.0, -1.0))
    assert not _is_horizontal((0.0, 1.0))


@pytest.mark.parametrize(
    "text",
    [
        "1 Introduction",
        "2 Related Work",
        "2.1 Dense Retrieval",
        "3.2.1 Experimental Setup",
        "4. Experimental Setup",
        "Appendix A Proofs",
    ],
)
def test_real_numbered_headings_are_detected(text):
    assert _is_heading(line(text), BODY)


@pytest.mark.parametrize(
    "text",
    [
        # All from arXiv:2610.30697, at exactly body size.
        "0.99 therefore misses about ten thousand true matches per",
        "0.9999 reduces them to about one hundred. Ranking qual-",
        "0.3378 for the exact neighbors (Section 4.2). Exact search",
        "0.979. Lychee answers 115,163 QPS at this operating point",
        "0.3352. Lychee matches 0.990 recall at 65,178 QPS with",
        # From arXiv:2610.30612 and 2610.32608.
        "1.24 after internal benchmarks [35], [36]; Vespa's tutorial",
        "2025 shared task developed that research setting [1, 2].",
        "21 worsened on Robust04). The fusion-QPP line [5], [6]",
        "1.28M",
    ],
)
def test_wrapped_body_lines_beginning_with_numbers_are_not_headings(text):
    """A section number is a small integer followed by a capitalised word.

    These lines are mid-sentence continuations that happen to start with a
    figure. Reading them as headings fragmented one 14-page paper into 60
    "sections" and scattered its body across them.
    """
    assert not _is_heading(line(text), BODY)


@pytest.mark.parametrize(
    "text,size",
    [
        ("Aleksandar Mihajlovic", 11.96),
        ("Yiteng Tu", 11.96),
        ("Beijing, China", 11.96),
        ("Linz, Austria", 11.96),
        ("Freya Wentzela, IAG Wilsond and Shuo Wanga", 11.96),
        ("University of Washington, Tacoma School of Engineering", 11.0),
    ],
)
def test_author_and_affiliation_lines_are_not_headings(text, size):
    """Short, capitalised, larger than body text -- indistinguishable from a
    heading by styling alone, which is why the size-only rule was removed."""
    assert not _is_heading(line(text, size=size), BODY)


@pytest.mark.parametrize(
    "text", ["Abstract", "Introduction", "Related Work", "Conclusions", "References",
             "Keywords", "CCS Concepts", "Limitations", "Acknowledgements"]
)
def test_conventional_unnumbered_headings_are_detected(text):
    assert _is_heading(line(text), BODY)


def test_a_long_line_is_never_a_heading_however_it_is_styled():
    long_line = " ".join(["Word"] * 20)
    assert not _is_heading(line(long_line, size=18.0), BODY)


@pytest.mark.parametrize(
    "text",
    [
        # arXiv:2610.30697 -- reads as section 26.10 followed by a capital.
        "26.10. The first server has one NVIDIA H100 GPU [41]",
        "1.24 after internal benchmarks [35], [36]; Vespa's tutorial",
        "3 Results are consistent with prior work [12, 14]",
    ],
)
def test_lines_containing_citations_are_not_headings(text):
    """A heading names a section; it does not cite. The bracketed-reference test
    catches the residue that survives the capitalisation rule."""
    assert not _is_heading(line(text), BODY)


@pytest.mark.parametrize("text", ["method", "evaluation.", "methods.", "results", "background"])
def test_lowercase_body_words_are_not_headings(text):
    """From arXiv:2610.30612 and 2610.32608: single lowercase words sitting on
    their own line in body text matched the conventional-heading list and
    produced sections named 'method' and 'evaluation.'."""
    assert not _is_heading(line(text), BODY)


@pytest.mark.parametrize("text", ["Method", "METHOD", "Evaluation", "ACKNOWLEDGMENT", "Abstract"])
def test_properly_cased_conventional_headings_still_pass(text):
    assert _is_heading(line(text), BODY)


def test_a_multi_line_title_is_joined(tmp_path):
    """arXiv titles wrap. Taking only the largest single line truncated
    'Directional Posterior Feedback Fusion for Query Reformulation in Dense
    Retrieval' down to 'in Dense Retrieval'."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(
        pymupdf.Rect(72, 60, 540, 130),
        "Directional Posterior Feedback Fusion\nfor Query Reformulation",
        fontsize=17.0,
        fontname="helv",
    )
    page.insert_textbox(
        pymupdf.Rect(72, 150, 540, 600),
        "1 Introduction\nThe body of the paper runs on for long enough to set "
        "the modal font size well below the title size used above.",
        fontsize=10,
        fontname="helv",
    )
    path = tmp_path / "wrapped-title.pdf"
    doc.save(path)
    doc.close()

    parsed = parse_pdf(path)
    assert parsed.title == "Directional Posterior Feedback Fusion for Query Reformulation"


def test_repeated_heading_names_get_distinct_section_paths(tmp_path):
    """A section_path addresses a citation, so it has to be unique within a
    document. arXiv:2610.32608 produced 'Method' three times."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(
        pymupdf.Rect(72, 60, 540, 700),
        "Method\nFirst description of the approach taken here.\n"
        "Method\nSecond description appearing later in the paper.\n"
        "Method\nThird description appearing later still in the text.",
        fontsize=10,
        fontname="helv",
    )
    path = tmp_path / "dupes.pdf"
    doc.save(path)
    doc.close()

    paths = [s.section_path for s in parse_pdf(path).sections]
    assert len(paths) == len(set(paths)), paths
    assert paths[0] == "Method"
    assert paths[1].startswith("Method")


# --------------------------------------------------------------------------
# Font weight.
#
# arXiv:2610.30612 is an IEEE Access paper whose headings are SMALLER than its
# body text -- 8.97pt NimbusSanL-Bold headings against a 10.0pt body -- so any
# size-based rule structurally cannot parse it, and it yielded 2 sections out of
# ~40 with its entire body unreachable. Weight is the discriminator, and it is
# also what separates a real lettered subsection from a reference entry.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["I. INTRODUCTION", "II. RELATED WORK", "V. RESULTS", "VII. CONCLUSION"],
)
def test_bold_roman_headings_smaller_than_body_are_detected(text):
    assert _is_heading(line(text, size=8.97, bold=True), BODY)


@pytest.mark.parametrize(
    "text",
    [
        "A. DENSE AND SPARSE RETRIEVAL",
        "B. HYBRID FUSION AND THE RRF DEFAULT",
        "D. QUERY PERFORMANCE PREDICTION",
        "H. DOES RERANKING PAY THE TAX?",
    ],
)
def test_bold_lettered_subsections_are_detected(text):
    assert _is_heading(line(text, size=8.97, bold=True), BODY)


@pytest.mark.parametrize(
    "text",
    [
        # Reference entries: author initials look exactly like lettered
        # subsections, and "L." also looks like a roman numeral.
        "L. Fei-Fei. ImageNet: A large-scale hierarchical image",
        "V. Ramanujan, Y. Bitton, K. Marathe, S. Mussmann,",
        "C. Author and D. Coauthor. Some paper title here",
    ],
)
def test_unbolded_reference_entries_are_not_headings(text):
    assert not _is_heading(line(text, size=7.58, bold=False), BODY)


@pytest.mark.parametrize("text", ["VOLUME 14, 2026", "AADITYA", "CHAUHAN", "TREC-COVID"])
def test_unbolded_page_furniture_is_not_a_heading(text):
    assert not _is_heading(line(text, size=5.98, bold=False), BODY)


def test_headings_after_the_reference_list_are_suppressed(tmp_path):
    """Everything after References is bibliography and author biographies.

    This is what produced sections named 'L. Fei-Fei. ImageNet: A large-scale
    hierarchical image' and a stray 'Method' at the end of two real papers.
    """
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(
        pymupdf.Rect(72, 60, 540, 700),
        "1 Introduction\nThe body of this paper is long enough to establish the "
        "modal body font size for the document as a whole.\n"
        "References\nSome bibliography text follows here.\n"
        "Method\nThis must not open a new section after the bibliography.",
        fontsize=10,
        fontname="helv",
    )
    path = tmp_path / "refs.pdf"
    doc.save(path)
    doc.close()

    paths = [s.section_path for s in parse_pdf(path).sections]
    assert "References" in paths
    assert "Method" not in paths


def test_an_appendix_after_references_is_still_a_section(tmp_path):
    """Appendices legitimately follow the bibliography."""
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_textbox(
        pymupdf.Rect(72, 60, 540, 700),
        "1 Introduction\nThe body of this paper is long enough to establish the "
        "modal body font size for the document as a whole.\n"
        "References\nSome bibliography text follows here.\n"
        "Appendix A Stress Set\nThe full stress set results are tabulated here.",
        fontsize=10,
        fontname="helv",
    )
    path = tmp_path / "appendix.pdf"
    doc.save(path)
    doc.close()

    paths = [s.section_path for s in parse_pdf(path).sections]
    assert any(p.startswith("Appendix") for p in paths), paths
