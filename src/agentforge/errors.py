"""Failure taxonomy.

AgentForge has no fallback paths. Every error raises, immediately, with no
degraded alternative -- because a silent fallback corrupts attribution. If the
reranker quietly no-ops on some queries, or ingestion drops the documents that
failed to parse, the reported retrieval recall is inflated and every measured
delta becomes unattributable.

The one distinction that matters, and the only place this rule needs care:

    A DEFECT is a bug in the system. It raises.
        Schema-invalid model output, Postgres unreachable, a PDF that will not
        parse, a chunk_id that is not in the index, missing config at startup,
        a reranker that will not load, a latency run without cache bypass.

    A MODEL-QUALITY OUTCOME is the phenomenon this system exists to measure.
        It is recorded as data and surfaced in the report.
        A fabricated quote (quote absent from its cited chunk), or a quote that
        does not entail its claim. These get a verdict row and ship flagged as
        unverified. Dropping them WOULD be a fallback: it would hide the failure
        the evaluation exists to count, and would quietly inflate groundedness.
"""


class AgentForgeError(Exception):
    """Base class. Nothing in this package raises a bare Exception."""


class ConfigError(AgentForgeError):
    """Configuration absent, malformed, or internally inconsistent at startup."""


class IngestError(AgentForgeError):
    """A document or chunk could not be processed.

    Aborts the whole ingest run. Partial ingests are forbidden: indexing 97% of
    a corpus and reporting recall over it silently drops exactly the hard
    documents, which inflates the number.
    """


class IndexingError(AgentForgeError):
    """An index could not be built, or is missing when a query needs it."""


class RetrievalError(AgentForgeError):
    """A retrieval stage failed, or returned a chunk_id absent from the corpus."""


class SchemaViolation(AgentForgeError):
    """A model returned output that does not satisfy its declared schema.

    Structured outputs are enforced API-side, so this is a defect rather than
    something to repair with a retry loop.
    """


class IncomparableRuns(AgentForgeError):
    """Two runs were compared whose configs differ in more than one field.

    A delta between them cannot be attributed to any single knob, so the harness
    refuses to tabulate it.
    """


class MeasurementError(AgentForgeError):
    """A metric was requested that the stored rows cannot honestly support.

    Raised for, among others: latency from a cache-backed run, p95 from fewer
    observations than the configured minimum, or kappa against reference labels
    that are still pending human review.
    """


class QuotaExhausted(AgentForgeError):
    """The provider's rate limit will not reset within a wait worth making.

    Short per-minute limits are waited out. A daily limit is not: the run stops
    here, and re-running after the reset replays every completed call from the
    cache for free, so nothing is lost and no question is silently skipped.
    """
