"""Structure-aware small-to-big chunking.

Chunking is not a retrieval tuning detail in this system: the chunk IS the
citation span. A claim's verbatim quote is matched against one chunk's text, so
chunk size sets the resolution of the entire groundedness story. 1024-token
chunks make a citation point at half a page and make the judge's entailment call
mushy; 256-token chunks keep attribution tight.

The cost of small chunks is context, and small-to-big pays it off: retrieve and
cite at ~256 tokens, hand the generator the expanded parent section. Precision
where it matters for ranking and attribution, context where it matters for
synthesis.

Two invariants the rest of the system depends on:

  * `document_text[chunk.char_start:chunk.char_end] == chunk.text`. This is what
    makes a citation resolve to a real location rather than a plausible
    reconstruction. Offsets are into the CANONICALISED document text (see
    ingest.normalize), because that is also what quotes are matched against.
  * `chunk.text` is verbatim source. The contextual prefix lives only in
    `embed_text`; baking it into `text` would make every verbatim quote fail.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from agentforge.errors import IngestError

TokenCounter = Callable[[str], int]

# Sentence boundary: terminal punctuation, then whitespace, then something that
# starts a new sentence. Deliberately conservative -- it under-splits on
# abbreviations ("et al. showed") rather than cutting mid-sentence, because a
# chunk that ends mid-clause reads badly as a citation.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


@dataclass(frozen=True)
class Section:
    """A contiguous run of text under one heading.

    `char_offset` is where this section's text begins in the whole document, so
    chunk offsets can be reported document-absolute.
    """

    section_path: str
    text: str
    page: int
    char_offset: int


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    section_path: str
    page: int
    char_start: int
    char_end: int
    text: str
    embed_text: str
    parent_text: str | None
    n_tokens: int


@dataclass(frozen=True)
class _Piece:
    """A sentence (or sentence fragment) with its offset inside its section."""

    text: str
    start: int
    tokens: int

    @property
    def end(self) -> int:
        return self.start + len(self.text)


def _split_sentences(text: str, counter: TokenCounter) -> list[_Piece]:
    pieces: list[_Piece] = []
    cursor = 0
    for part in _SENTENCE_END.split(text):
        if not part:
            continue
        start = text.index(part, cursor)
        pieces.append(_Piece(text=part, start=start, tokens=counter(part)))
        cursor = start + len(part)
    return pieces


def _split_oversized(piece: _Piece, budget: int, counter: TokenCounter) -> list[_Piece]:
    """Break a single sentence that exceeds the budget, on word boundaries.

    Dropping it would make part of a paper permanently unretrievable, so it is
    split even though the result may read mid-clause.
    """
    out: list[_Piece] = []
    words = piece.text.split()
    current: list[str] = []
    cursor = piece.start
    for word in words:
        candidate = [*current, word]
        if current and counter(" ".join(candidate)) > budget:
            text = " ".join(current)
            start = piece.text.index(text, cursor - piece.start) + piece.start
            out.append(_Piece(text=text, start=start, tokens=counter(text)))
            cursor = start + len(text)
            current = [word]
        else:
            current = candidate
    if current:
        text = " ".join(current)
        start = piece.text.index(text, cursor - piece.start) + piece.start
        out.append(_Piece(text=text, start=start, tokens=counter(text)))
    return out


def _group(pieces: list[_Piece], budget: int, overlap: int) -> list[list[_Piece]]:
    """Greedily pack sentences up to the budget, carrying `overlap` tokens back."""
    groups: list[list[_Piece]] = []
    current: list[_Piece] = []
    used = 0

    for piece in pieces:
        if current and used + piece.tokens > budget:
            groups.append(current)
            carried: list[_Piece] = []
            carried_tokens = 0
            for previous in reversed(current):
                if carried_tokens + previous.tokens > overlap:
                    break
                carried.insert(0, previous)
                carried_tokens += previous.tokens
            current = [*carried, piece]
            used = carried_tokens + piece.tokens
        else:
            current.append(piece)
            used += piece.tokens

    if current:
        groups.append(current)
    return groups


def chunk_sections(
    sections: list[Section],
    *,
    title: str,
    token_counter: TokenCounter,
    chunk_tokens: int = 256,
    overlap_tokens: int = 38,
    contextual_prefix: bool = True,
    parent_section: bool = True,
) -> list[Chunk]:
    """Chunk a document's sections. Never spans a section boundary.

    `token_counter` is required rather than defaulted: a counter that disagrees
    with the embedding model's tokenizer would silently move every boundary, and
    chunk size is a measured arm.
    """
    if chunk_tokens <= 0:
        raise IngestError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if overlap_tokens >= chunk_tokens:
        raise IngestError(
            f"overlap_tokens ({overlap_tokens}) must be smaller than chunk_tokens "
            f"({chunk_tokens}), or chunking cannot advance"
        )

    chunks: list[Chunk] = []
    ordinal = 0

    for section in sections:
        if not section.text.strip():
            raise IngestError(
                f"section {section.section_path!r} is empty; a silently dropped "
                "section is a silently unretrievable part of the document"
            )

        pieces: list[_Piece] = []
        for piece in _split_sentences(section.text, token_counter):
            if piece.tokens > chunk_tokens:
                pieces.extend(_split_oversized(piece, chunk_tokens, token_counter))
            else:
                pieces.append(piece)

        for group in _group(pieces, chunk_tokens, overlap_tokens):
            start = group[0].start
            end = group[-1].end
            text = section.text[start:end]
            prefix = f"{title}. {section.section_path}. " if contextual_prefix else ""
            chunks.append(
                Chunk(
                    ordinal=ordinal,
                    section_path=section.section_path,
                    page=section.page,
                    char_start=section.char_offset + start,
                    char_end=section.char_offset + end,
                    text=text,
                    embed_text=f"{prefix}{text}",
                    parent_text=section.text if parent_section else None,
                    n_tokens=token_counter(text),
                )
            )
            ordinal += 1

    if not chunks:
        raise IngestError("no chunks produced: the document has no sections")
    return chunks
