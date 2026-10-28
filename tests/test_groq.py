"""Tests for the Groq transport, with the HTTP layer scripted.

No test here touches the network: the project runs on a free-tier quota, and
what needs pinning is the handling around the call -- strict schemas, a
per-minute 429 waited out, a daily 429 stopped on, truncation refused.
"""

from __future__ import annotations

import json

import pytest

from agentforge.config import MODEL_PRICES, MODEL_PROVIDERS
from agentforge.errors import ConfigError, QuotaExhausted, SchemaViolation
from agentforge.llm import groq_transport, transport_for

SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
    "additionalProperties": False,
}


def ok(content='{"verdict": "entailed"}', finish="stop"):
    body = {
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 7},
    }
    return 200, {}, json.dumps(body).encode()


class ScriptedPost:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.bodies: list[dict] = []

    def __call__(self, url, headers, body):
        self.bodies.append(json.loads(body))
        return self.responses.pop(0)


def call(transport):
    return transport(
        model="openai/gpt-oss-20b", system="sys",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=256, schema=SCHEMA, thinking={"reasoning_effort": "low"},
    )


def test_request_uses_strict_schema_and_low_reasoning():
    post = ScriptedPost(ok())
    result = call(groq_transport("k", post=post))
    sent = post.bodies[0]
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["response_format"]["json_schema"]["schema"] == SCHEMA
    assert sent["reasoning_effort"] == "low"
    assert sent["messages"][0] == {"role": "system", "content": "sys"}
    assert result == {"parsed": {"verdict": "entailed"}, "input_tokens": 40, "output_tokens": 7}


def test_a_per_minute_limit_is_waited_out_then_retried():
    waits: list[float] = []
    post = ScriptedPost((429, {"Retry-After": "3"}, b""), ok())
    result = call(groq_transport("k", sleep=waits.append, post=post))
    assert waits == [3.5]
    assert result["parsed"] == {"verdict": "entailed"}


def test_a_daily_limit_stops_the_run_instead_of_waiting_hours():
    post = ScriptedPost((429, {"retry-after": "5400"}, b""))
    with pytest.raises(QuotaExhausted, match="daily quota"):
        call(groq_transport("k", sleep=lambda s: None, post=post))


def test_truncated_output_is_refused_not_parsed():
    post = ScriptedPost(ok(content='{"verd', finish="length"))
    with pytest.raises(SchemaViolation, match="max_tokens"):
        call(groq_transport("k", post=post))


def test_a_non_200_raises_with_the_body():
    post = ScriptedPost((400, {}, b'{"error": "schema not supported"}'))
    with pytest.raises(SchemaViolation, match="HTTP 400"):
        call(groq_transport("k", post=post))


def test_every_priced_model_has_a_provider_and_unknown_models_raise():
    assert MODEL_PROVIDERS.keys() == MODEL_PRICES.keys()
    with pytest.raises(ConfigError, match="no provider"):
        transport_for("gpt-5")


def test_a_missing_key_raises():
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        groq_transport("")
