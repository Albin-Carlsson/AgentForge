"""Canonical text normalisation.

This module is small and has more tests than anything else in the project,
because it sits on the seam where a real bug can masquerade as a hallucination.

A claim's citation carries a verbatim quote, and fabrication detection is an
exact substring check of that quote against the cited chunk's text. PDF
extraction genuinely produces ligatures, soft hyphens at line breaks,
non-breaking spaces and curly quotes, so an unnormalised comparison would report
fabricated quotes that are in fact faithful -- inflating the headline
hallucination number with parser artefacts.

The rule: normalise once, canonically, on BOTH sides -- the stored chunk text
and the model's quote. Any mismatch that survives is a genuine fabrication, and
is treated as data (see errors.py), not repaired.
"""

from __future__ import annotations

import re
import unicodedata

# Ligatures that PDF fonts emit as single codepoints. NFKC handles most, but it
# also rewrites things we want left alone (superscripts in citations, for one),
# so the ligatures are mapped explicitly and NFKC is not used.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "œ": "oe", "Œ": "OE", "æ": "ae", "Æ": "AE",
}

# Quote and dash variants. A model asked for a verbatim quote will frequently
# normalise these itself, so the stored side has to meet it halfway.
_PUNCT = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    "…": "...",
}

# Zero-width and formatting characters that carry no textual meaning.
_INVISIBLE = dict.fromkeys(
    [
        "\u00ad",  # soft hyphen -- the PDF line-break artefact
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\u2060",  # word joiner
        "\ufeff",  # BOM / zero-width no-break space
        "\u200e",  # left-to-right mark
        "\u200f",  # right-to-left mark
    ],
    "",
)

_WHITESPACE = re.compile(r"\s+")

# A hyphen followed by a line break is a word split across lines: "retriev-\nal".
# Joining it is lossy in the rare case of a genuine end-of-line hyphenated
# compound, which is accepted -- the alternative leaves a fake token in the text
# that no quote can ever match.
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")

_TRANSLATION = str.maketrans({**_LIGATURES, **_PUNCT, **_INVISIBLE})


def canonical(text: str) -> str:
    """Normalise text for quote matching. Idempotent.

    Applied identically to stored chunk text and to model-supplied quotes.
    """
    text = unicodedata.normalize("NFC", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = text.translate(_TRANSLATION)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def quote_found(quote: str, chunk_text: str) -> bool:
    """Deterministic fabrication check: `str.find`, no model, no variance.

    An empty quote is not a citation, so it cannot be found -- a model emitting
    one has failed to cite, which is a model-quality outcome and gets recorded.
    """
    needle = canonical(quote)
    if not needle:
        return False
    return needle in canonical(chunk_text)
