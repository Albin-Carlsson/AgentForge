"""Tests for the cached LLM client.

Every test here runs without an API key and without a network call, because the
behaviour worth testing is not "does Anthropic reply" but:

  * is a repeated call free and identical (content-addressed cache),
  * does changing one parameter actually invalidate the cache,
  * is cost computed from measured tokens at list price,
  * does a schema-invalid response raise rather than get repaired,
  * is latency refused when it came from a cache hit.

The transport is injected, so the cache and accounting logic are tested in
isolation from the SDK.
"""

from __future__ import annotations

import pytest

from agentforge.errors import ConfigError, MeasurementError, SchemaViolation
from agentforge.llm import LLMClient, LLMResponse, cache_key, price


class FakeTransport:
    """Records calls and replays scripted responses."""

    def __init__(self, replies: list[dict] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls: list[dict] = []

    def __call__(self, *, model, system, messages, max_tokens, schema, thinking):
        self.calls.append(
            {
                "model": model,
                "system": system,
                "messages": messages,
                "max_tokens": max_tokens,
                "schema": schema,
                "thinking": thinking,
            }
        )
        if self.replies:
            return self.replies.pop(0)
        return {"parsed": {"ok": True}, "input_tokens": 100, "output_tokens": 20}


SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def client(transport, **kwargs) -> LLMClient:
    return LLMClient(model="claude-haiku-4-5", transport=transport, **kwargs)


def test_price_is_computed_from_measured_tokens_at_list():
    # haiku 4.5: $1.00/MTok in, $5.00/MTok out
    assert price("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=0) == 1.0
    assert price("claude-haiku-4-5", input_tokens=0, output_tokens=1_000_000) == 5.0
    assert price("claude-haiku-4-5", input_tokens=1000, output_tokens=200) == pytest.approx(
        0.001 + 0.001
    )


def test_an_unpriced_model_raises_rather_than_reporting_zero_cost():
    with pytest.raises(ConfigError, match="no list price"):
        price("some-unreleased-model", 10, 10)


def test_cache_key_depends_on_every_input_that_changes_the_output():
    base = {
        "model": "claude-haiku-4-5",
        "system": "s",
        "messages": [{"role": "user", "content": "q"}],
        "max_tokens": 100,
        "schema": SCHEMA,
        "thinking": None,
    }
    key = cache_key(**base)
    assert cache_key(**base) == key, "the key must be stable"
    assert cache_key(**{**base, "model": "claude-sonnet-5"}) != key
    assert cache_key(**{**base, "system": "other"}) != key
    assert cache_key(**{**base, "max_tokens": 101}) != key
    assert cache_key(**{**base, "messages": [{"role": "user", "content": "other"}]}) != key
    assert cache_key(**{**base, "thinking": {"type": "enabled", "budget_tokens": 1024}}) != key


def test_cache_key_is_insensitive_to_dict_ordering():
    """Otherwise an identical request would miss the cache at random."""
    a = cache_key(
        model="m", system="s", messages=[{"role": "user", "content": "q"}],
        max_tokens=1, schema={"a": 1, "b": 2}, thinking=None,
    )
    b = cache_key(
        model="m", system="s", messages=[{"role": "user", "content": "q"}],
        max_tokens=1, schema={"b": 2, "a": 1}, thinking=None,
    )
    assert a == b


def test_a_repeated_call_hits_the_cache_and_costs_nothing(db):
    transport = FakeTransport()
    llm = client(transport)
    first = llm.complete("system", "question", SCHEMA)
    second = llm.complete("system", "question", SCHEMA)

    assert len(transport.calls) == 1, "the second call must not reach the API"
    assert first.parsed == second.parsed
    assert first.cached is False and second.cached is True
    assert second.cost_usd == 0.0, "a cache hit is free and must be reported as free"


def test_changing_the_prompt_misses_the_cache(db):
    transport = FakeTransport()
    llm = client(transport)
    llm.complete("system", "question one", SCHEMA)
    llm.complete("system", "question two", SCHEMA)
    assert len(transport.calls) == 2


def test_cache_bypass_always_calls_through(db):
    transport = FakeTransport()
    llm = client(transport, cache_bypass=True)
    llm.complete("system", "question", SCHEMA)
    response = llm.complete("system", "question", SCHEMA)
    assert len(transport.calls) == 2
    assert response.cached is False


def test_latency_from_a_cache_hit_is_refused(db):
    """A cached call returns in microseconds; reporting that as latency is fiction."""
    transport = FakeTransport()
    llm = client(transport)
    llm.complete("system", "question", SCHEMA)
    cached = llm.complete("system", "question", SCHEMA)
    assert cached.cached is True
    with pytest.raises(MeasurementError, match="cache"):
        _ = cached.require_latency_ms()


def test_latency_is_available_from_a_real_call(db):
    llm = client(FakeTransport(), cache_bypass=True)
    response = llm.complete("system", "question", SCHEMA)
    assert response.require_latency_ms() >= 0


def test_a_response_missing_a_required_field_raises(db):
    transport = FakeTransport([{"parsed": {}, "input_tokens": 1, "output_tokens": 1}])
    with pytest.raises(SchemaViolation, match="ok"):
        client(transport).complete("system", "question", SCHEMA)


def test_a_response_with_an_unexpected_field_raises(db):
    transport = FakeTransport(
        [{"parsed": {"ok": True, "surprise": 1}, "input_tokens": 1, "output_tokens": 1}]
    )
    with pytest.raises(SchemaViolation, match="surprise"):
        client(transport).complete("system", "question", SCHEMA)


def test_a_wrongly_typed_field_raises(db):
    transport = FakeTransport(
        [{"parsed": {"ok": "yes"}, "input_tokens": 1, "output_tokens": 1}]
    )
    with pytest.raises(SchemaViolation, match="ok"):
        client(transport).complete("system", "question", SCHEMA)


def test_usage_is_accumulated_across_calls(db):
    transport = FakeTransport(
        [
            {"parsed": {"ok": True}, "input_tokens": 100, "output_tokens": 10},
            {"parsed": {"ok": False}, "input_tokens": 200, "output_tokens": 20},
        ]
    )
    llm = client(transport)
    llm.complete("s", "q1", SCHEMA)
    llm.complete("s", "q2", SCHEMA)
    assert llm.usage.input_tokens == 300
    assert llm.usage.output_tokens == 30
    assert llm.usage.calls == 2
    assert llm.usage.cost_usd == pytest.approx(price("claude-haiku-4-5", 300, 30))


def test_cache_hits_do_not_inflate_accumulated_cost(db):
    transport = FakeTransport()
    llm = client(transport)
    llm.complete("s", "q", SCHEMA)
    billed = llm.usage.cost_usd
    llm.complete("s", "q", SCHEMA)
    assert llm.usage.cost_usd == billed
    assert llm.usage.cache_hits == 1


def test_an_absent_api_key_raises_only_when_a_real_call_is_needed():
    """M1 and M2 run without a key; the error must name that boundary."""
    from agentforge.llm import default_transport

    # An empty key, not None: None reads .env, which may hold a real key.
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        default_transport(api_key="")


def test_response_reports_which_model_answered(db):
    response = client(FakeTransport()).complete("s", "q", SCHEMA)
    assert isinstance(response, LLMResponse)
    assert response.model == "claude-haiku-4-5"


def test_anthropic_effort_goes_in_output_config_not_thinking(monkeypatch):
    import types

    import anthropic

    from agentforge.llm import _thinking_for, default_transport

    sent: dict = {}

    class Messages:
        def create(self, **kwargs):
            sent.update(kwargs)
            return types.SimpleNamespace(
                stop_reason="end_turn",
                content=[types.SimpleNamespace(type="text", text='{"ok": true}')],
                usage=types.SimpleNamespace(input_tokens=5, output_tokens=2),
            )

    monkeypatch.setattr(anthropic, "Anthropic", lambda api_key: types.SimpleNamespace(messages=Messages()))
    thinking = _thinking_for("claude-haiku-5-5", "low")
    result = default_transport("k")(
        model="claude-haiku-5-5", system="s", messages=[{"role": "user", "content": "q"}],
        max_tokens=64, schema=SCHEMA, thinking=thinking,
    )
    assert sent["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "low"}
    assert "thinking" not in sent
    assert result["parsed"] == {"ok": True}
    assert _thinking_for("openai/gpt-oss-20b", "low") == {"reasoning_effort": "low"}
