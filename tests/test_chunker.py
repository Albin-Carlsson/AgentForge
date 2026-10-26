"""Tests for structure-aware small-to-big chunking.

Chunking is not a retrieval tuning detail here: the chunk IS the citation span,
so these tests are really about whether a claim can be resolved back to the
exact text that supports it.

The invariant that matters most is the offset invariant --
`document_text[chunk.char_start:chunk.char_end] == chunk.text` -- because that
is what makes a citation point at a real location in a real document rather than
at a plausible-looking reconstruction.
"""

from __future__ import annotations

import itertools

import pytest

from agentforge.errors import IngestError
from agentforge.ingest.chunker import Section, chunk_sections


def words(text: str) -> int:
    """A deterministic stand-in for the model tokenizer."""
    return len(text.split())


def build_document(sections: list[Section]) -> str:
    """Reconstruct the document text these sections were sliced from."""
    out: list[str] = []
    cursor = 0
    for section in sorted(sections, key=lambda s: s.char_offset):
        out.append(" " * (section.char_offset - cursor))
        out.append(section.text)
        cursor = section.char_offset + len(section.text)
    return "".join(out)


def make_sections() -> list[Section]:
    intro = "We study hybrid retrieval. " * 12
    method = "The reranker rescores candidates. " * 12
    return [
        Section(section_path="1. Introduction", text=intro.strip(), page=1, char_offset=0),
        Section(
            section_path="2. Method",
            text=method.strip(),
            page=3,
            char_offset=len(intro.strip()) + 10,
        ),
    ]


def chunk(sections, **overrides):
    kwargs = {
        "title": "Hybrid Retrieval for Science",
        "token_counter": words,
        "chunk_tokens": 20,
        "overlap_tokens": 4,
        "contextual_prefix": True,
        "parent_section": True,
    }
    return chunk_sections(sections, **(kwargs | overrides))


def test_a_token_counter_must_be_supplied_explicitly():
    """No default heuristic: a counter that disagrees with the embedding model's
    tokenizer would silently change every chunk boundary."""
    with pytest.raises(TypeError):
        chunk_sections(make_sections(), title="t")  # type: ignore[call-arg]


def test_chunks_never_span_two_sections():
    chunks = chunk(make_sections())
    assert len(chunks) > 2
    for c in chunks:
        assert c.section_path in {"1. Introduction", "2. Method"}
    # No chunk's text may contain content from both sections.
    for c in chunks:
        assert not ("hybrid retrieval" in c.text.lower() and "rescores" in c.text.lower())


def test_offsets_resolve_to_the_exact_source_text():
    """The invariant a citation depends on."""
    sections = make_sections()
    document = build_document(sections)
    for c in chunk(sections):
        assert document[c.char_start : c.char_end] == c.text, c.section_path


def test_ordinals_are_contiguous_across_the_document():
    chunks = chunk(make_sections())
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunks_respect_the_token_target():
    for c in chunk(make_sections(), chunk_tokens=20):
        assert c.n_tokens <= 20, c.text


def test_consecutive_chunks_in_a_section_overlap():
    chunks = [c for c in chunk(make_sections(), overlap_tokens=6) if c.section_path.startswith("1.")]
    assert len(chunks) >= 2
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words, "overlap should carry context across the boundary"


def test_zero_overlap_produces_disjoint_spans():
    chunks = [c for c in chunk(make_sections(), overlap_tokens=0) if c.section_path.startswith("1.")]
    assert len(chunks) >= 2
    for earlier, later in itertools.pairwise(chunks):
        assert earlier.char_end <= later.char_start


def test_a_short_section_becomes_one_chunk():
    sections = [Section(section_path="3. Conclusion", text="We conclude.", page=9, char_offset=0)]
    chunks = chunk(sections)
    assert len(chunks) == 1
    assert chunks[0].text == "We conclude."
    assert chunks[0].page == 9


def test_embed_text_carries_the_contextual_prefix():
    c = chunk(make_sections())[0]
    assert c.embed_text.startswith("Hybrid Retrieval for Science")
    assert "1. Introduction" in c.embed_text
    assert c.text in c.embed_text
    # The verbatim text must stay clean: quotes are matched against `text`, and a
    # prefix baked into it would make every quote fail.
    assert not c.text.startswith("Hybrid Retrieval for Science")


def test_contextual_prefix_can_be_disabled_as_an_arm():
    c = chunk(make_sections(), contextual_prefix=False)[0]
    assert c.embed_text == c.text


def test_parent_text_is_the_whole_section_for_small_to_big():
    sections = make_sections()
    chunks = [c for c in chunk(sections) if c.section_path == "2. Method"]
    assert all(c.parent_text == sections[1].text for c in chunks)
    assert len(chunks[0].parent_text) > len(chunks[0].text)


def test_parent_text_is_absent_when_the_arm_is_off():
    c = chunk(make_sections(), parent_section=False)[0]
    assert c.parent_text is None


def test_an_empty_section_raises_rather_than_being_skipped():
    """A silently dropped section is a silently unretrievable part of a paper."""
    sections = [Section(section_path="4. Empty", text="   ", page=2, char_offset=0)]
    with pytest.raises(IngestError, match="empty"):
        chunk(sections)


def test_a_sentence_longer_than_the_budget_is_split_not_dropped():
    long_sentence = " ".join(f"w{i}" for i in range(60))
    sections = [Section(section_path="5. Long", text=long_sentence, page=1, char_offset=0)]
    chunks = chunk(sections, chunk_tokens=20)
    assert len(chunks) >= 3
    assert all(c.n_tokens <= 20 for c in chunks)
    # Nothing may be lost: every word survives somewhere.
    seen = {w for c in chunks for w in c.text.split()}
    assert seen == set(long_sentence.split())


def test_splitting_prefers_sentence_boundaries():
    text = "First sentence here. Second sentence here. Third sentence here."
    sections = [Section(section_path="6. S", text=text, page=1, char_offset=0)]
    chunks = chunk(sections, chunk_tokens=7, overlap_tokens=0)
    assert all(c.text.endswith(".") for c in chunks), [c.text for c in chunks]
