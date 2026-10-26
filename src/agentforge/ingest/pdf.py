"""PDF parsing with PyMuPDF.

Chosen over the heavyweight layout stack (Nougat, `unstructured`, full GROBID)
for two reasons: no model weights to ship, and character-level geometry. Every
text block comes back with a bounding box, so a chunk's character range can be
turned into page rectangles -- which is what lets a citation resolve to a
highlightable region rather than a filename.

Extraction is per LINE, not per block. PyMuPDF returns one block per layout
region, and in ACM-style two-column papers a section heading and the paragraph
beneath it land in the same block -- so block-level heading detection saw a
40-word blob and rejected it, and arXiv:2610.30612 parsed as only
['ACKNOWLEDGMENT', 'REFERENCES'] with its entire body silently unretrievable.
Lines also give tighter bounding boxes, so a citation highlights a few lines
rather than a whole column.

Heading detection uses font WEIGHT as much as size. That is not a stylistic
preference: arXiv:2610.30612 is an IEEE Access paper whose headings are smaller
than its body text (8.97pt bold against a 10.0pt body), so a size-based rule
parsed 2 sections out of ~40 and left the entire body unreachable. Weight is
also what separates a real lettered subsection ("A. DATASETS", bold) from a
reference entry whose author initial looks identical ("L. Fei-Fei.", not bold).

Headings are also suppressed after the bibliography, because reference entries
and author biographies otherwise read as sections.

Detection is a heuristic regardless, so it is That is a heuristic, so it is MEASURED rather than trusted: see
`ingest.fidelity`, which scores section-boundary recall against ar5iv's HTML
rendering of the same paper. A parser whose error rate is unknown makes every
downstream retrieval number unknown too.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

import pymupdf

from agentforge.errors import IngestError
from agentforge.ingest.chunker import Section
from agentforge.ingest.normalize import canonical

PARSER_VERSION = f"pymupdf-{pymupdf.__doc__.split()[1] if pymupdf.__doc__ else '1'}"

# "1 Introduction", "2.1 Ablations", "IV. Results", "Appendix A Proofs".
#
# Three constraints, each one learned from a real false positive:
#   * the first component is 1-99, never 0 -- "0.99 therefore misses..." is a
#     wrapped body line, not section zero point ninety-nine;
#   * each sub-component is at most two digits -- "0.979. Lychee answers" and
#     "0.3352. Lychee matches" are measurements;
#   * the number must be followed by a CAPITALISED word. This is the strongest
#     discriminator by far: real headings name something ("2 Related Work"),
#     while wrapped prose continues in lower case ("1.24 after internal
#     benchmarks", "2025 shared task developed", "21 worsened on Robust04").
_NUMBERED_HEADING = re.compile(
    r"^\s*(?:[1-9]\d?(?:\.\d{1,2}){0,2}\.?|[IVXLC]{1,5}\.|Appendix\s+[A-Z])\s+[A-Z]"
)
# Unnumbered but conventional headings in scientific papers.
_NAMED_HEADING = re.compile(
    r"^\s*(abstract|introduction|related work|background|method(?:s|ology)?|"
    r"experiments?|results?|evaluation|discussion|conclusions?|"
    r"limitations?|references|acknowledge?ments?|keywords?|ccs concepts|"
    r"future work|appendix|ethics statement|reproducibility statement)\s*[:.]?\s*$",
    re.IGNORECASE,
)

# "A. DATASETS", "H. DOES RERANKING PAY THE TAX?" -- IEEE-style lettered
# subsections. Indistinguishable from a reference entry's author initial
# ("L. Fei-Fei. ImageNet: ...") except by weight, so these REQUIRE bold.
_LETTERED_HEADING = re.compile(r"^\s*[A-Z]\.\s+[A-Z]")

# "[41]", "[35], [36]", "[12, 14]" -- a heading names a section, it does not cite.
_CITATION = re.compile(r"\[\s*\d")

_REFERENCES = re.compile(r"^\s*(references|bibliography)\s*[:.]?\s*$", re.IGNORECASE)
_APPENDIX = re.compile(r"^\s*appendix\b", re.IGNORECASE)

# PyMuPDF span flags: bit 4 (value 16) is bold. The font name is checked too,
# because subsetted Type1 faces such as NimbusSanL-Bold do not always set it.
_BOLD_FLAG = 1 << 4


def _is_bold(span: dict) -> bool:
    if int(span.get("flags", 0)) & _BOLD_FLAG:
        return True
    return "bold" in str(span.get("font", "")).lower()

_MAX_HEADING_WORDS = 14


def _is_horizontal(direction: tuple[float, float]) -> bool:
    """Whether a line of text runs left-to-right.

    arXiv stamps the left margin of every PDF with a rotated identifier, and on
    arXiv:2610.29464 that stamp is 20.0pt against the paper's own 17.22pt title
    -- so selecting the largest text on page one picked the stamp. Rotated text
    in a paper is margin furniture, never content.
    """
    return abs(direction[0]) > abs(direction[1])


@dataclass(frozen=True)
class TextSpan:
    """One extracted block, with where it lives in the text and on the page."""

    char_start: int
    char_end: int
    page: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class ParsedDocument:
    title: str | None
    text: str
    sections: list[Section]
    spans: list[TextSpan]
    n_pages: int
    parser_version: str


@dataclass
class _Line:
    text: str
    size: float
    page: int
    bold: bool
    bbox: tuple[float, float, float, float]


def _read_lines(path: pathlib.Path) -> tuple[list[_Line], int]:
    try:
        document = pymupdf.open(path)
    except Exception as exc:  # pymupdf raises bare Exception subclasses
        raise IngestError(f"cannot open PDF {path}: {exc}") from exc

    lines: list[_Line] = []
    try:
        n_pages = document.page_count
        if n_pages == 0:
            raise IngestError(f"cannot open PDF {path}: it has no pages")
        for index in range(n_pages):
            page = document.load_page(index)
            payload = page.get_text("dict")
            for block in payload.get("blocks", []):
                if block.get("type") != 0:  # 0 == text; images are skipped
                    continue
                for line in block.get("lines", []):
                    if not _is_horizontal(tuple(line.get("dir", (1.0, 0.0)))):
                        continue
                    spans = line.get("spans", [])
                    text = canonical("".join(s.get("text", "") for s in spans))
                    if not text:
                        continue
                    sizes = [float(s.get("size", 0.0)) for s in spans]
                    lines.append(
                        _Line(
                            text=text,
                            size=max(sizes) if sizes else 0.0,
                            page=index + 1,
                            bold=any(_is_bold(s) for s in spans),
                            bbox=tuple(float(v) for v in line["bbox"]),  # type: ignore[arg-type]
                        )
                    )
    finally:
        document.close()

    if not lines:
        raise IngestError(f"cannot open PDF {path}: no extractable text")
    return lines, n_pages


def _body_size(lines: list[_Line]) -> float:
    """Modal font size, weighted by text length: the body text size."""
    weights: dict[float, int] = {}
    for line in lines:
        key = round(line.size, 1)
        weights[key] = weights.get(key, 0) + len(line.text)
    return max(weights.items(), key=lambda kv: kv[1])[0]


def _is_heading(line: _Line, body_size: float) -> bool:
    """A heading is a short line that is either larger than body text, or
    numbered, or one of the conventional section names.

    Short is the operative constraint: at line granularity a heading occupies
    its own line, so a long line is prose regardless of how it is styled.
    """
    if len(line.text.split()) > _MAX_HEADING_WORDS:
        return False
    if _CITATION.search(line.text):
        return False

    # Bold counts as prominence in its own right, which is what makes papers
    # with sub-body-size headings parseable at all.
    prominent = line.bold or line.size > body_size + 0.75
    at_least_body = line.size >= body_size - 0.25

    if _NUMBERED_HEADING.match(line.text):
        return prominent or at_least_body
    if _LETTERED_HEADING.match(line.text):
        return line.bold
    # A conventional name only counts when it is cased like a heading. Bare
    # lowercase words sitting on their own line in body text ("method",
    # "evaluation.") otherwise became sections named after themselves.
    if _NAMED_HEADING.match(line.text) and line.text[:1].isupper():
        return prominent or at_least_body

    # No size-only rule. A short, capitalised, larger-than-body line is
    # indistinguishable from an author name or affiliation ("Yiteng Tu",
    # "Beijing, China", both 11.96pt against 9.96pt body), and admitting them
    # invented sections that scattered real body text across them. Unnumbered
    # headings are recognised by name instead.
    return False


def parse_pdf(path: pathlib.Path | str) -> ParsedDocument:
    """Parse a PDF into canonical text, sections, and a character-to-page map."""
    path = pathlib.Path(path)
    if not path.exists():
        raise IngestError(f"cannot open PDF {path}: file does not exist")

    lines, n_pages = _read_lines(path)
    body_size = _body_size(lines)

    # The title is the largest text on page one.
    first_page = [line for line in lines if line.page == 1]
    title_line = max(first_page, key=lambda line: (line.size, -len(line.text)), default=None)
    # Titles wrap, so the title is every CONSECUTIVE page-one line at the
    # largest size -- taking the single biggest line truncated "Directional
    # Posterior Feedback Fusion for Query Reformulation in Dense Retrieval" to
    # "in Dense Retrieval". Disqualified only by an actual heading pattern, not
    # by being large: the title is by definition the biggest text on page one.
    title: str | None = None
    title_lines: list[_Line] = []
    if (
        title_line is not None
        and title_line.size > body_size
        and not _NUMBERED_HEADING.match(title_line.text)
        and not _NAMED_HEADING.match(title_line.text)
    ):
        anchor = first_page.index(title_line)
        title_lines = [title_line]
        for step in (-1, 1):
            cursor = anchor + step
            while 0 <= cursor < len(first_page):
                candidate = first_page[cursor]
                if abs(candidate.size - title_line.size) > 0.3 or _is_heading(
                    candidate, body_size
                ):
                    break
                title_lines.append(candidate)
                cursor += step
        title_lines.sort(key=lambda line: first_page.index(line))
        title = " ".join(line.text for line in title_lines)

    # Assemble the document text, recording each line's span as we go.
    #
    # Lines are joined with "\n" rather than " " on purpose: a word hyphenated
    # across a line break ("retriev-\nal") is only rejoinable by
    # normalize.canonical while that newline is still there. Quote matching
    # canonicalises both sides, so the join happens at comparison time.
    text_parts: list[str] = []
    spans: list[TextSpan] = []
    cursor = 0
    placed: list[tuple[_Line, int, int]] = []
    for line in lines:
        start = cursor
        text_parts.append(line.text)
        cursor += len(line.text)
        spans.append(
            TextSpan(char_start=start, char_end=cursor, page=line.page, bbox=line.bbox)
        )
        placed.append((line, start, cursor))
        text_parts.append("\n")
        cursor += 1
    text = "".join(text_parts)

    # Walk the blocks, opening a new section at each heading.
    sections: list[Section] = []
    seen_paths: dict[str, int] = {}
    current_path: str | None = None
    current_page = 1
    body_start: int | None = None
    body_end: int | None = None

    def close() -> None:
        nonlocal current_path, body_start, body_end
        if current_path is not None and body_start is not None and body_end is not None:
            body = text[body_start:body_end]
            if body.strip():
                # A repeated heading name would make two different spans share
                # one address, so occurrences after the first are numbered.
                seen_paths[current_path] = seen_paths.get(current_path, 0) + 1
                occurrence = seen_paths[current_path]
                unique_path = (
                    current_path if occurrence == 1 else f"{current_path} ({occurrence})"
                )
                sections.append(
                    Section(
                        section_path=unique_path,
                        text=body,
                        page=current_page,
                        char_offset=body_start,
                    )
                )
        body_start = body_end = None

    title_line_ids = {id(line) for line in title_lines}
    in_bibliography = False
    for line, start, end in placed:
        if id(line) in title_line_ids:
            continue
        if _is_heading(line, body_size):
            # Past the bibliography, only an appendix opens a section: reference
            # entries and author biographies otherwise become sections named
            # after whoever is cited.
            if in_bibliography and not _APPENDIX.match(line.text):
                continue
            close()
            current_path = line.text
            current_page = line.page
            if _REFERENCES.match(line.text):
                in_bibliography = True
            elif _APPENDIX.match(line.text):
                in_bibliography = False
            continue
        if current_path is None:
            continue  # front matter before the first heading
        if body_start is None:
            body_start = start
        body_end = end
    close()

    if not sections:
        raise IngestError(
            f"{path.name}: no sections detected. Heading detection is font-size based; "
            "a paper-shaped PDF with no detectable headings means the parser failed "
            "rather than that the document has no structure."
        )

    return ParsedDocument(
        title=title,
        text=text,
        sections=sections,
        spans=spans,
        n_pages=n_pages,
        parser_version=PARSER_VERSION,
    )


def bbox_for(
    spans: list[TextSpan], char_start: int, char_end: int
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Page rectangles covering a character range, one union per page.

    This is what turns a chunk's offsets into something a reader can be pointed
    at: `page=7, bbox=(72, 430, 540, 512)`.
    """
    by_page: dict[int, list[tuple[float, float, float, float]]] = {}
    for span in spans:
        if span.char_end <= char_start or span.char_start >= char_end:
            continue
        by_page.setdefault(span.page, []).append(span.bbox)

    located: list[tuple[int, tuple[float, float, float, float]]] = []
    for page in sorted(by_page):
        boxes = by_page[page]
        located.append(
            (
                page,
                (
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes),
                ),
            )
        )
    return located
