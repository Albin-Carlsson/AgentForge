"""Tests for observability.

Langfuse earns its place here for one concrete reason: the per-stage spans ARE
the latency decomposition the README reports. It is not a logo in a tech-stack
list, it is where a number comes from.

Which also means its failure modes matter. Partial configuration raises rather
than silently tracing nothing, because "we have tracing" and "we have half a
Langfuse key" look identical until you go looking for the spans.
"""

from __future__ import annotations

import pytest

from agentforge.errors import ConfigError
from agentforge.obs import NullObserver, RecordingObserver, build_observer


def test_an_unconfigured_environment_yields_a_null_observer():
    """Tracing is optional; the retrieval tier runs with no accounts at all."""
    observer = build_observer(public_key=None, secret_key=None, host="h")
    assert isinstance(observer, NullObserver)
    assert observer.enabled is False


@pytest.mark.parametrize(
    "public,secret",
    [("pk-only", None), (None, "sk-only")],
)
def test_half_configured_credentials_raise(public, secret):
    """Silently tracing nothing is worse than not tracing: the README quotes
    stage latencies that come from these spans."""
    with pytest.raises(ConfigError, match="LANGFUSE"):
        build_observer(public_key=public, secret_key=secret, host="h")


def test_the_null_observer_accepts_every_call_without_effect():
    observer = NullObserver()
    with observer.span("retrieve", corpus="x"):
        pass
    observer.record_generation(model="m", input_tokens=1, output_tokens=2, cost_usd=0.1,
                               cached=False, latency_ms=3.0, stage="s")
    observer.record_score(name="groundedness", value=1.0)
    observer.flush()


def test_the_recording_observer_captures_spans_in_order():
    observer = RecordingObserver()
    with observer.span("plan"):
        pass
    with observer.span("research", n=2):
        pass
    assert [s["name"] for s in observer.spans] == ["plan", "research"]
    assert observer.spans[1]["metadata"] == {"n": 2}
    assert all(s["duration_ms"] >= 0 for s in observer.spans)


def test_a_span_records_its_duration_even_when_the_body_raises():
    """A failed stage's latency is still data about the failure."""
    observer = RecordingObserver()
    with pytest.raises(RuntimeError), observer.span("boom"):
        raise RuntimeError("x")
    assert observer.spans[0]["name"] == "boom"
    assert observer.spans[0]["error"] == "RuntimeError: x"


def test_generations_and_scores_are_recorded_separately():
    observer = RecordingObserver()
    observer.record_generation(model="claude-haiku-4-5", input_tokens=100, output_tokens=20,
                               cost_usd=0.0002, cached=False, latency_ms=12.0, stage="synthesize")
    observer.record_score(name="groundedness", value=0.83, comment="3 of 4")
    assert observer.generations[0]["model"] == "claude-haiku-4-5"
    assert observer.generations[0]["stage"] == "synthesize"
    assert observer.scores[0] == {"name": "groundedness", "value": 0.83, "comment": "3 of 4"}


def test_a_cached_generation_is_marked_so_traces_do_not_imply_model_time():
    observer = RecordingObserver()
    observer.record_generation(model="m", input_tokens=1, output_tokens=1, cost_usd=0.0,
                               cached=True, latency_ms=0.05, stage="judge")
    assert observer.generations[0]["cached"] is True


def test_the_llm_client_reports_every_call_to_the_observer(db):
    from agentforge.llm import LLMClient

    def transport(*, model, system, messages, max_tokens, schema, thinking):
        return {"parsed": {"ok": True}, "input_tokens": 10, "output_tokens": 2}

    observer = RecordingObserver()
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    client = LLMClient(model="claude-haiku-4-5", transport=transport, observer=observer)
    client.complete("sys", "prompt", schema)
    client.complete("sys", "prompt", schema)  # cache hit

    assert len(observer.generations) == 2
    assert observer.generations[0]["cached"] is False
    assert observer.generations[1]["cached"] is True
    assert observer.generations[1]["cost_usd"] == 0.0
