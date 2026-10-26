"""Tests for the quote-matching seam.

These matter more than they look: a false negative here is reported as a
fabricated quote, which would inflate the project's headline hallucination rate
with PDF parser artefacts.
"""

import pytest

from agentforge.ingest.normalize import canonical, quote_found


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("the ﬁrst retriever", "the first retriever"),          # fi ligature
        ("re­trieval", "retrieval"),                             # soft hyphen
        ("retriev-\nal augmented", "retrieval augmented"),            # hyphen line break
        ("“quoted”", '"quoted"'),                           # curly quotes
        ("don’t", "don't"),                                      # curly apostrophe
        ("a—b", "a-b"),                                          # em dash
        ("spaced out", "spaced out"),                            # NBSP
        ("multi   \n\t space", "multi space"),                        # whitespace collapse
        ("  padded  ", "padded"),                                     # strip
        ("zero\u200bwidth", "zerowidth"),                             # ZWSP removed
    ],
)
def test_canonical(raw, expected):
    assert canonical(raw) == expected


def test_canonical_is_idempotent():
    raw = "  the ﬁrst  re­triev-\nal “model”—yes  "
    once = canonical(raw)
    assert canonical(once) == once


def test_quote_found_across_parser_artefacts():
    chunk = "We evaluate the ﬁrst-stage re­triever on BEIR’s SciFact split."
    assert quote_found("the first-stage retriever on BEIR's SciFact split", chunk)


def test_quote_found_tolerates_whitespace_reflow():
    chunk = "Hybrid retrieval\n   combines BM25   and dense scores."
    assert quote_found("Hybrid retrieval combines BM25 and dense scores.", chunk)


def test_fabricated_quote_is_not_found():
    chunk = "We evaluate on SciFact."
    assert not quote_found("We evaluate on HotpotQA.", chunk)


def test_empty_quote_is_never_found():
    assert not quote_found("", "any text at all")
    assert not quote_found("   ", "any text at all")


def test_partial_word_overlap_is_still_a_match_only_if_substring():
    chunk = "cross-encoder reranking improved nDCG@10 by six points"
    assert quote_found("reranking improved nDCG@10", chunk)
    assert not quote_found("reranking improved nDCG@20", chunk)
