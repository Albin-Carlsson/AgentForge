"""Observability.

Langfuse is here for one concrete reason: the per-stage spans ARE the latency
decomposition the README reports. "end-to-end 6.1 s, of which retrieval 180 ms
and rerank 40 ms" is a statement about where time goes, and these spans are
where it comes from. That makes it a measurement component, not a logo in a
tech-stack list.

The division of responsibility is deliberate. Postgres is the system of record
for everything quoted as a result -- runs, per-question rows, claim verdicts,
kappa -- because a headline number should not depend on a hosted service being
reachable. Langfuse owns what it is genuinely better at: per-span traces, token
and cost attribution, and a UI for clicking into one bad answer. Claim-level
scores are dual-written so a trace can be filtered by groundedness.

Tracing is optional: the retrieval tier and ingestion run with no accounts of
any kind. PARTIAL configuration, though, raises -- "we have tracing" and "we
have half a Langfuse key" look identical right up until you go looking for the
spans that were never recorded.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Protocol

from agentforge.errors import ConfigError


class Observer(Protocol):
    enabled: bool

    def span(self, name: str, **metadata: Any): ...

    def record_generation(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cached: bool,
        latency_ms: float,
        stage: str,
    ) -> None: ...

    def record_score(self, *, name: str, value: float, comment: str | None = None) -> None: ...

    def flush(self) -> None: ...


class NullObserver:
    """No tracing configured. Accepts every call and does nothing."""

    enabled = False

    @contextmanager
    def span(self, name: str, **metadata: Any):
        yield

    def record_generation(self, **_: Any) -> None:
        return None

    def record_score(self, **_: Any) -> None:
        return None

    def flush(self) -> None:
        return None


class RecordingObserver:
    """In-memory observer. Used by tests, and useful for a local run."""

    enabled = True

    def __init__(self) -> None:
        self.spans: list[dict[str, Any]] = []
        self.generations: list[dict[str, Any]] = []
        self.scores: list[dict[str, Any]] = []

    @contextmanager
    def span(self, name: str, **metadata: Any):
        started = time.perf_counter()
        record: dict[str, Any] = {"name": name, "metadata": metadata, "error": None}
        self.spans.append(record)
        try:
            yield
        except Exception as exc:
            # A failed stage's latency is still data about the failure.
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["duration_ms"] = (time.perf_counter() - started) * 1000

    def record_generation(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cached: bool,
        latency_ms: float,
        stage: str,
    ) -> None:
        self.generations.append(
            {
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                "cached": cached,
                "latency_ms": latency_ms,
                "stage": stage,
            }
        )

    def record_score(self, *, name: str, value: float, comment: str | None = None) -> None:
        self.scores.append({"name": name, "value": value, "comment": comment})

    def flush(self) -> None:
        return None


class LangfuseObserver:
    """Traces to Langfuse Cloud.

    Self-hosting Langfuse v3 means Postgres plus ClickHouse plus Redis plus
    MinIO -- four services to trace a side project. The cloud free tier is an
    SDK and two environment variables, and the bullet is equally true.
    """

    enabled = True

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        from langfuse import Langfuse

        self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)

    @contextmanager
    def span(self, name: str, **metadata: Any):
        started = time.perf_counter()
        error: str | None = None
        try:
            yield
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._client.create_event(
                name=name,
                metadata={
                    **metadata,
                    "duration_ms": (time.perf_counter() - started) * 1000,
                    "error": error,
                },
            )

    def record_generation(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        cached: bool,
        latency_ms: float,
        stage: str,
    ) -> None:
        self._client.create_event(
            name=f"generation:{stage}",
            metadata={
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost_usd,
                # Marked, so a trace's timings are never mistaken for model time.
                "cached": cached,
                "latency_ms": latency_ms,
            },
        )

    def record_score(self, *, name: str, value: float, comment: str | None = None) -> None:
        self._client.create_score(name=name, value=value, comment=comment)

    def flush(self) -> None:
        self._client.flush()


def build_observer(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
) -> Observer:
    """Pick an observer from configuration. Half-configured credentials raise."""
    if public_key is None and secret_key is None and host is None:
        from agentforge.settings import settings

        config = settings()
        public_key = config.langfuse_public_key
        secret_key = config.langfuse_secret_key
        host = config.langfuse_host

    if bool(public_key) != bool(secret_key):
        missing = "LANGFUSE_SECRET_KEY" if public_key else "LANGFUSE_PUBLIC_KEY"
        raise ConfigError(
            f"{missing} is unset while its counterpart is set. Refusing to run with "
            "tracing half-configured: the stage latencies reported in the README come "
            "from these spans, and silently recording none of them is worse than "
            "recording none deliberately. Unset both to disable tracing."
        )

    if not public_key:
        return NullObserver()
    return LangfuseObserver(public_key, secret_key or "", host or "https://cloud.langfuse.com")
