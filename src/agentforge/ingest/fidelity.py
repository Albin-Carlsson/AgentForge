"""Scoring the PDF parser against ar5iv HTML.

Heading detection is a font-geometry heuristic, and the previous module's
history shows how badly it can fail on individual papers. The question an
interviewer asks about any heuristic is "how often is it wrong?", and the only
acceptable answer is a number.

arXiv publishes a LaTeXML rendering of each paper's source at
arxiv.org/html/<id>, so its `<h2 class="ltx_title_section">` tags are ground
truth derived from the author's own sectioning rather than from how the PDF
happens to look. Scoring against it costs nothing and needs no labelling.

ar5iv was the obvious candidate and turned out to be the wrong one: it has no
rendering for recent papers and redirects to the abstract page rather than
returning 404, so a naive check scores every recent paper as unmeasurable. That
is detected here by the absence of section-title classes.

Two measurements, deliberately separate:

  recall     -- of the sections the author wrote, how many did the parser find?
                A miss means part of a paper is unretrievable.
  precision  -- of the sections the parser reported, how many are real?
                A spurious heading fragments a section and scatters its body.

A paper with no ar5iv rendering is a COVERAGE gap, reported as such. It is not a
parser failure, and scoring it zero would understate the parser; both metrics
are None in that case and the caller reports how many papers lacked ground
truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from agentforge.errors import IngestError
from agentforge.ingest.normalize import canonical

# LaTeXML's classes for real headings. Front and back matter count: the first
# run scored "Abstract", "Keywords" and "References" as spurious parser output
# on every paper, because they carry their own classes rather than
# ltx_title_section. They are sections, and the parser is right to find them.
# ltx_title_document (the paper title) and float captions stay excluded.
_SECTION_TITLE_CLASSES = (
    "ltx_title_section",
    "ltx_title_subsection",
    "ltx_title_subsubsection",
    "ltx_title_appendix",
    "ltx_title_abstract",
    "ltx_title_bibliography",
    "ltx_title_acknowledgements",
    "ltx_title_keywords",
)

# \paragraph{} produces a RUN-IN heading: bold text at the start of a paragraph,
# on the same line as the prose that follows it. A line-based PDF parser cannot
# see these as headings, by construction.
#
# They are excluded from the recall denominator and counted separately, because
# including them measures something the parser never claimed to do. On
# arXiv:2610.32318 they are 25 of 38 ground-truth headings, which alone dragged
# that paper's apparent recall to 0.158 -- a number that would have described
# the metric's definition rather than the parser's quality.
_RUN_IN_TITLE_CLASS = "ltx_title_paragraph"

# Leading section labels, in every style the real corpus produced. Each
# alternative below exists because a correct parse scored as a miss without it:
#
#   1  2.1  3.3.          decimal, with or without a trailing period
#   I  II  IV.            roman, WITH OR WITHOUT a period -- LaTeXML emits
#                         "I Introduction" while the PDF renders "I. INTRODUCTION"
#   II-A  V-C             compound roman-letter, LaTeXML's subsection form
#   A.  A.1.  E.2.        lettered, period required
#   Appendix A
#
# A bare capital with no period is NOT a label: "A Model-Agnostic Framework"
# must keep its article.
_LEADING_LABEL = re.compile(
    r"""^\s*(?:
          [Aa]ppendix\s+[A-Z]\b
        | [0-9]+(?:\.[0-9]+)*\.?
        | [IVXLC]{1,5}(?:-[A-Z0-9]+)+\.?
        | [IVXLC]{1,5}\.?(?=\s)
        | [A-Z]\.(?:[0-9]+\.?)*
      )\s+""",
    re.VERBOSE,
)
_PUNCTUATION = re.compile(r"[^\w\s]+")


@dataclass(frozen=True)
class FidelityScores:
    recall: float | None
    precision: float | None
    n_truth: int
    n_found: int
    n_run_in: int = 0  # run-in \paragraph headings, excluded from scoring
    matched: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    spurious: list[str] = field(default_factory=list)


class _LatexmlHeadings(HTMLParser):
    def __init__(self, classes: tuple[str, ...] = _SECTION_TITLE_CLASSES) -> None:
        super().__init__(convert_charrefs=True)
        self._classes = classes
        self.headings: list[str] = []
        self._depth = 0
        self._buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._depth:
            self._depth += 1  # nested <span class="ltx_tag"> holding the number
            return
        classes = dict(attrs).get("class") or ""
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6") and any(
            name in classes for name in self._classes
        ):
            self._depth = 1
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._depth:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._depth:
            return
        self._depth -= 1
        if self._depth == 0:
            heading = canonical("".join(self._buffer))
            if heading:
                self.headings.append(heading)


def run_in_headings_from_latexml(html: str) -> list[str]:
    r"""Run-in \paragraph headings: real structure, invisible to a line parser."""
    parser = _LatexmlHeadings(classes=(_RUN_IN_TITLE_CLASS,))
    try:
        parser.feed(html)
    except Exception as exc:
        raise IngestError(f"could not parse LaTeXML HTML: {exc}") from exc
    return parser.headings


def headings_from_latexml(html: str) -> list[str]:
    """Block-level section headings, in document order, from a LaTeXML rendering."""
    parser = _LatexmlHeadings()
    try:
        parser.feed(html)
    except Exception as exc:
        raise IngestError(f"could not parse LaTeXML HTML: {exc}") from exc
    return parser.headings


def normalise_heading(heading: str) -> str:
    """Strip numbering, punctuation and case so two renderings are comparable.

    The PDF and the HTML frequently number the same section differently -- one
    paper's "3 Methodology" is the other's "II. METHODOLOGY" -- so comparing
    raw strings would score a correct parse as a total miss.
    """
    text = _LEADING_LABEL.sub("", canonical(heading))
    text = _PUNCTUATION.sub(" ", text)
    return " ".join(text.lower().split())


def score_sections(found: list[str], truth: list[str]) -> FidelityScores:
    """Compare detected section headings against ar5iv's."""
    truth_keys = {normalise_heading(t): t for t in truth if normalise_heading(t)}
    found_keys = {normalise_heading(f): f for f in found if normalise_heading(f)}

    if not truth_keys:
        return FidelityScores(
            recall=None, precision=None, n_truth=0, n_found=len(found_keys)
        )

    matched = sorted(truth_keys.keys() & found_keys.keys())
    missed = sorted(truth_keys.keys() - found_keys.keys())
    spurious = sorted(found_keys.keys() - truth_keys.keys())

    return FidelityScores(
        recall=len(matched) / len(truth_keys),
        precision=len(matched) / len(found_keys) if found_keys else 0.0,
        n_truth=len(truth_keys),
        n_found=len(found_keys),
        matched=[truth_keys[k] for k in matched],
        missed=[truth_keys[k] for k in missed],
        spurious=[found_keys[k] for k in spurious],
    )


def text_coverage(sections: list, n_chars: int) -> float:
    """Fraction of the document's characters that fall inside some section.

    The companion metric to section recall, and the one that actually bounds
    retrieval quality. A missed heading merges its body into the previous
    section: the section_path on a citation gets coarser, but no text becomes
    unretrievable. Text that falls outside every section is genuinely lost --
    never chunked, never embedded, never citable.

    Expect a value below 1.0 even on a perfect parse: the title, authors and
    affiliations precede the first heading and belong to no section. Read it as
    a floor to watch, not a target to hit.
    """
    if n_chars <= 0:
        raise ValueError("coverage of an empty document is undefined")
    covered = sum(len(section.text) for section in sections)
    return min(covered / n_chars, 1.0)


def fetch_latexml_html(arxiv_id: str) -> str | None:
    """Fetch arXiv's LaTeXML rendering, or None when there is none.

    None is a coverage gap, not an error in the thing being measured: not every
    paper has a rendering, and the caller reports how many lacked ground truth.
    """
    import urllib.error

    from agentforge.ingest.arxiv import ARXIV_HTML, _throttled_get

    try:
        html = _throttled_get(f"{ARXIV_HTML}/{arxiv_id}").decode("utf-8", errors="replace")
    except IngestError as exc:
        if "404" in str(exc):
            return None
        raise
    except urllib.error.HTTPError as exc:  # pragma: no cover - defensive
        if exc.code == 404:
            return None
        raise
    # A response carrying no section-title classes is not a rendering: both
    # arXiv and ar5iv serve the abstract page for papers they have not built.
    return html if any(name in html for name in _SECTION_TITLE_CLASSES) else None
