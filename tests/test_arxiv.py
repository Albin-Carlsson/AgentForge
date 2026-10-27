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


# --------------------------------------------------------------------------
# Retry classification.
#
# The 300-paper ingest died at paper 111 on ConnectionResetError (errno 54),
# which the retry did not catch: it handled urllib.error.URLError and
# TimeoutError, but a reset-by-peer arrives as a bare OSError. A 404, by
# contrast, must NOT be retried -- it is an answer, not a failure.
# --------------------------------------------------------------------------


class _Opener:
    """A stand-in for urlopen that replays a scripted sequence of outcomes."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, request, timeout=None, context=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)


class _Response:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_connection_reset_is_retried_and_then_succeeds(monkeypatch):
    from agentforge.ingest import arxiv as mod

    monkeypatch.setattr(mod, "MIN_REQUEST_INTERVAL_S", 0.0)
    monkeypatch.setattr(mod, "_RETRY_BACKOFF_S", 0.0)
    opener = _Opener([ConnectionResetError(54, "Connection reset by peer"), b"ok"])
    assert mod._throttled_get("http://example.test/x", opener=opener) == b"ok"
    assert opener.calls == 2


def test_exhausted_retries_raise_with_the_cause(monkeypatch):
    from agentforge.ingest import arxiv as mod

    monkeypatch.setattr(mod, "MIN_REQUEST_INTERVAL_S", 0.0)
    monkeypatch.setattr(mod, "_RETRY_BACKOFF_S", 0.0)
    opener = _Opener([ConnectionResetError(54, "reset")] * mod.MAX_ATTEMPTS)
    with pytest.raises(IngestError, match="after 3 attempts"):
        mod._throttled_get("http://example.test/x", opener=opener)
    assert opener.calls == mod.MAX_ATTEMPTS


def test_a_404_is_not_retried(monkeypatch):
    """A 404 is an answer. Retrying it wastes three rate-limited slots."""
    import urllib.error

    from agentforge.ingest import arxiv as mod

    monkeypatch.setattr(mod, "MIN_REQUEST_INTERVAL_S", 0.0)
    monkeypatch.setattr(mod, "_RETRY_BACKOFF_S", 0.0)
    error = urllib.error.HTTPError("http://example.test/x", 404, "Not Found", {}, None)
    opener = _Opener([error])
    with pytest.raises(IngestError, match="404"):
        mod._throttled_get("http://example.test/x", opener=opener)
    assert opener.calls == 1


def test_a_429_is_retried(monkeypatch):
    """Rate limiting is transient by definition."""
    import urllib.error

    from agentforge.ingest import arxiv as mod

    monkeypatch.setattr(mod, "MIN_REQUEST_INTERVAL_S", 0.0)
    monkeypatch.setattr(mod, "_RETRY_BACKOFF_S", 0.0)
    opener = _Opener(
        [urllib.error.HTTPError("http://e.test/x", 429, "Too Many", {}, None), b"ok"]
    )
    assert mod._throttled_get("http://e.test/x", opener=opener) == b"ok"
    assert opener.calls == 2


def test_a_500_is_retried(monkeypatch):
    import urllib.error

    from agentforge.ingest import arxiv as mod

    monkeypatch.setattr(mod, "MIN_REQUEST_INTERVAL_S", 0.0)
    monkeypatch.setattr(mod, "_RETRY_BACKOFF_S", 0.0)
    opener = _Opener(
        [urllib.error.HTTPError("http://e.test/x", 503, "Unavailable", {}, None), b"ok"]
    )
    assert mod._throttled_get("http://e.test/x", opener=opener) == b"ok"
