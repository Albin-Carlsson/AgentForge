"""Tests for arXiv metadata parsing.

Parsing is tested against a canned Atom feed so it needs no network; the two
network functions are thin wrappers over `urllib` and are exercised by the
actual ingest.

`published` is load-bearing rather than decorative: contamination-freedom for
the end-to-end tier rests on every paper post-dating the generator model's
training cutoff, so the date is stored per document and reported as a range.
"""

from __future__ import annotations

import datetime as dt

import pytest

from agentforge.errors import IngestError
from agentforge.ingest.arxiv import PaperMeta, ar5iv_url, parse_atom

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2609.01234v2</id>
    <updated>2026-09-15T10:00:00Z</updated>
    <published>2026-09-02T08:30:00Z</published>
    <title>Hybrid Retrieval with
      Reciprocal Rank Fusion</title>
    <summary>  We study the fusion of lexical and dense retrieval.
    </summary>
    <author><name>A. Researcher</name></author>
    <author><name>B. Coauthor</name></author>
    <link href="http://arxiv.org/abs/2609.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2609.01234v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.IR"/>
    <category term="cs.IR"/>
    <category term="cs.CL"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2610.05678v1</id>
    <updated>2026-10-01T00:00:00Z</updated>
    <published>2026-10-01T00:00:00Z</published>
    <title>Cross-Encoder Reranking Revisited</title>
    <summary>On when reranking helps.</summary>
    <author><name>C. Author</name></author>
    <link href="http://arxiv.org/abs/2610.05678v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2610.05678v1" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
  </entry>
</feed>
"""


def test_parses_every_entry():
    papers = parse_atom(FEED)
    assert len(papers) == 2
    assert all(isinstance(p, PaperMeta) for p in papers)


def test_identifier_keeps_the_version_and_exposes_the_base():
    paper = parse_atom(FEED)[0]
    assert paper.arxiv_id == "2609.01234v2"
    assert paper.base_id == "2609.01234"


def test_title_whitespace_is_reflowed():
    """arXiv wraps titles across lines; a stored title must be one line."""
    assert parse_atom(FEED)[0].title == "Hybrid Retrieval with Reciprocal Rank Fusion"


def test_published_date_is_parsed():
    paper = parse_atom(FEED)[0]
    assert paper.published == dt.date(2026, 9, 2)
    assert paper.updated == dt.date(2026, 9, 15)


def test_authors_and_categories_are_collected_in_order():
    paper = parse_atom(FEED)[0]
    assert paper.authors == ["A. Researcher", "B. Coauthor"]
    assert paper.categories == ["cs.IR", "cs.CL"]
    assert paper.primary_category == "cs.IR"


def test_pdf_and_abstract_urls_are_extracted():
    paper = parse_atom(FEED)[0]
    assert paper.pdf_url.endswith("/pdf/2609.01234v2")
    assert paper.abs_url.endswith("/abs/2609.01234v2")


def test_ar5iv_url_is_derived_from_the_base_id():
    """ar5iv's HTML is the ground truth the PDF parser is measured against."""
    assert ar5iv_url("2609.01234") == "https://ar5iv.labs.arxiv.org/html/2609.01234"
    assert parse_atom(FEED)[0].ar5iv_url == "https://ar5iv.labs.arxiv.org/html/2609.01234"


def test_malformed_xml_raises():
    with pytest.raises(IngestError, match="could not parse"):
        parse_atom("<feed><entry>unclosed")


def test_an_entry_without_a_pdf_link_raises_rather_than_being_skipped():
    """A silently skipped paper is a silently smaller corpus."""
    broken = FEED.replace(
        '<link title="pdf" href="http://arxiv.org/pdf/2609.01234v2" rel="related" type="application/pdf"/>',
        "",
    )
    with pytest.raises(IngestError, match="no PDF link"):
        parse_atom(broken)


def test_an_empty_feed_raises():
    empty = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    with pytest.raises(IngestError, match="no entries"):
        parse_atom(empty)


def test_tls_verification_is_enabled_against_a_real_ca_bundle():
    """Verification stays on; the bundle is named because the framework build
    ships an empty trust store. The tempting alternative is to disable
    verification, which this test exists to prevent regressing into."""
    import ssl

    from agentforge.ingest.arxiv import _ssl_context

    context = _ssl_context()
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert context.cert_store_stats()["x509_ca"] > 0
