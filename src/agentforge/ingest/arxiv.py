"""arXiv metadata and PDF retrieval.

The corpus is deliberately narrow -- retrieval and RAG papers in cs.IR/cs.CL --
for one overriding reason: the expensive part of evaluation is labels, and
writing labels requires understanding the documents. In a domain you know that
is an afternoon; in an unfamiliar one it is a month and the labels are subtly
wrong, which is worse than no labels because every reported number inherits the
error.

`published` is stored per document because contamination-freedom depends on it.
The end-to-end tier's claim to measure retrieval rather than recall-from-
pretraining rests on every paper post-dating the generator's training cutoff, so
the ingest reports its date range rather than asserting the property.

arXiv asks for at most one request every three seconds. That is enforced here
rather than left to the caller.
"""

from __future__ import annotations

import datetime as dt
import http.client
import pathlib
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache

import certifi

from agentforge.errors import IngestError

API = "http://export.arxiv.org/api/query"
AR5IV = "https://ar5iv.labs.arxiv.org/html"
ARXIV_HTML = "https://arxiv.org/html"
USER_AGENT = "AgentForge/0.1 (research corpus builder; contact via repo)"

# arXiv's stated rate limit. Not negotiable, and not the caller's problem.
MIN_REQUEST_INTERVAL_S = 3.0
_last_request_at = 0.0

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}
_VERSION_SUFFIX = re.compile(r"v\d+$")


@dataclass(frozen=True)
class PaperMeta:
    arxiv_id: str
    title: str
    authors: list[str]
    published: dt.date
    updated: dt.date
    abstract: str
    categories: list[str]
    primary_category: str
    pdf_url: str
    abs_url: str

    @property
    def base_id(self) -> str:
        return _VERSION_SUFFIX.sub("", self.arxiv_id)

    @property
    def ar5iv_url(self) -> str:
        return ar5iv_url(self.base_id)

    @property
    def html_url(self) -> str:
        """arXiv's own LaTeXML rendering -- the parser's ground truth.

        Preferred over ar5iv, which has no rendering for recent papers and
        silently redirects to the abstract page instead of 404ing.
        """
        return f"{ARXIV_HTML}/{self.arxiv_id}"


def ar5iv_url(base_id: str) -> str:
    """ar5iv renders arXiv LaTeX as structured HTML.

    Used as ground truth for section boundaries when scoring the PDF parser --
    see ingest.fidelity. It is a measurement aid, not a retrieval source.
    """
    return f"{AR5IV}/{base_id}"


def _reflow(text: str | None) -> str:
    return " ".join((text or "").split())


def _date(text: str | None, field: str) -> dt.date:
    if not text:
        raise IngestError(f"entry is missing {field}")
    try:
        return dt.datetime.fromisoformat(text).date()  # 3.11+ accepts a 'Z' suffix
    except ValueError as exc:
        raise IngestError(f"unparseable {field}: {text!r}") from exc


def parse_atom(xml_text: str) -> list[PaperMeta]:
    """Parse an arXiv Atom feed. Raises on anything it cannot fully interpret."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise IngestError(f"could not parse arXiv Atom feed: {exc}") from exc

    entries = root.findall("atom:entry", _NS)
    if not entries:
        raise IngestError("arXiv feed contains no entries")

    papers: list[PaperMeta] = []
    for entry in entries:
        raw_id = _reflow(entry.findtext("atom:id", namespaces=_NS))
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id
        if not arxiv_id:
            raise IngestError("entry is missing an id")

        pdf_url = abs_url = None
        for link in entry.findall("atom:link", _NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href")
            elif link.get("rel") == "alternate":
                abs_url = link.get("href")
        if not pdf_url:
            raise IngestError(
                f"arXiv entry {arxiv_id} has no PDF link; refusing to skip it, because "
                "a silently skipped paper is a silently smaller corpus"
            )

        categories = [
            c.get("term", "")
            for c in entry.findall("atom:category", _NS)
            if c.get("term")
        ]
        primary = entry.find("arxiv:primary_category", _NS)

        papers.append(
            PaperMeta(
                arxiv_id=arxiv_id,
                title=_reflow(entry.findtext("atom:title", namespaces=_NS)),
                authors=[
                    _reflow(a.findtext("atom:name", namespaces=_NS))
                    for a in entry.findall("atom:author", _NS)
                ],
                published=_date(entry.findtext("atom:published", namespaces=_NS), "published"),
                updated=_date(entry.findtext("atom:updated", namespaces=_NS), "updated"),
                abstract=_reflow(entry.findtext("atom:summary", namespaces=_NS)),
                categories=categories,
                primary_category=(primary.get("term") if primary is not None else "")
                or (categories[0] if categories else ""),
                pdf_url=pdf_url,
                abs_url=abs_url or f"http://arxiv.org/abs/{arxiv_id}",
            )
        )
    return papers


@lru_cache(maxsize=1)
def _ssl_context() -> ssl.SSLContext:
    """Verify TLS against certifi's bundle explicitly.

    The python.org framework build ships with an empty trust store until its
    "Install Certificates.command" is run, so a default context fails with
    CERTIFICATE_VERIFY_FAILED. Naming the bundle keeps the fix inside the repo
    instead of depending on a step someone has to remember on a new machine --
    and verification stays ON, which is the point: the alternative people reach
    for is disabling it.
    """
    return ssl.create_default_context(cafile=certifi.where())


MAX_ATTEMPTS = 3
_RETRY_BACKOFF_S = 2.0

# Retried: anything transient. ConnectionResetError (errno 54) killed the first
# 300-paper ingest at paper 111 -- it arrives as a bare OSError, not a URLError,
# so the original narrow catch missed it. URLError and TimeoutError are
# themselves OSError subclasses, so one clause covers all three.
_TRANSIENT = (OSError, http.client.HTTPException)

# Not retried: an HTTP status that is an ANSWER rather than a failure. Retrying
# a 404 just burns three rate-limited slots to learn the same thing.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _throttled_get(url: str, timeout: float = 60.0, opener=None) -> bytes:
    """Rate-limited GET with bounded retries on transient network failures.

    Retrying is resilience, not a fallback: there is no degraded path, and after
    the final attempt this raises with the underlying cause attached. The
    retries exist because a 300-paper ingest over a flaky network would
    otherwise abort most of the way through -- which it did, at paper 111 -- and
    the no-partial-ingest rule means an abort costs the whole run.

    `opener` is injectable so retry classification can be tested without a
    network: the distinction between "retry this" and "this is an answer"
    is logic, and logic deserves tests.
    """
    global _last_request_at
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    fetch = opener or (
        lambda req, timeout, context: urllib.request.urlopen(
            req, timeout=timeout, context=context
        )
    )
    last: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        wait = MIN_REQUEST_INTERVAL_S - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        try:
            with fetch(request, timeout=timeout, context=_ssl_context()) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE_STATUS:
                raise IngestError(
                    f"arXiv request for {url} returned HTTP {exc.code}: {exc.reason}"
                ) from exc
            last = exc
            time.sleep(_RETRY_BACKOFF_S * attempt)
        except _TRANSIENT as exc:
            last = exc
            time.sleep(_RETRY_BACKOFF_S * attempt)
        finally:
            _last_request_at = time.monotonic()

    raise IngestError(
        f"arXiv request failed for {url} after {MAX_ATTEMPTS} attempts: {last}"
    ) from last


def search(
    query: str, max_results: int, start: int = 0, sort_by: str = "submittedDate"
) -> list[PaperMeta]:
    """One page of search results. arXiv caps a page at 2000; be kinder than that."""
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
    )
    return parse_atom(_throttled_get(f"{API}?{params}").decode("utf-8"))


def download_pdf(paper: PaperMeta, directory: pathlib.Path) -> pathlib.Path:
    """Fetch a paper's PDF. Cached on disk by arxiv id; never re-downloaded."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{paper.arxiv_id}.pdf"
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    payload = _throttled_get(paper.pdf_url)
    if not payload.startswith(b"%PDF"):
        raise IngestError(
            f"{paper.arxiv_id}: downloaded {len(payload)} bytes that are not a PDF "
            f"(arXiv may have served an error page)"
        )
    destination.write_bytes(payload)
    return destination
