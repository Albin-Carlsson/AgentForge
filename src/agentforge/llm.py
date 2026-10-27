"""Anthropic client with a content-addressed cache and honest accounting.

Three jobs, all of which exist to protect the numbers rather than to be clever:

**Caching.** Every call is keyed on a hash of everything that could change its
output -- model, system prompt, messages, max_tokens, schema, thinking config.
A repeated call is free and byte-identical, so re-running an experiment costs
nothing and a config change re-bills only the calls it actually changed. That is
what makes an ablation over a 40-question tier affordable, and what makes
iterating on the judge prompt possible at all.

**Accounting.** Token counts are measured, not estimated, and the dollar column
is computed from the published rate card. A cache hit is reported as costing
zero, because it does, and it does not accumulate into the run's cost.

**Latency honesty.** A cache hit returns in microseconds. Asking a cached
response for its latency raises, because reporting that as a p95 would be
fiction -- the harness has the same guard at the run level, and this is the same
rule one layer down.

Structured output is enforced API-side via `output_config.format`, and validated
again here. A schema violation raises: there is no repair path, because a model
that returned the wrong shape is a defect, not a measurement.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentforge.config import MODEL_PRICES
from agentforge.db.conn import pool
from agentforge.errors import ConfigError, MeasurementError, SchemaViolation


def price(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD from measured token counts at list price."""
    if model not in MODEL_PRICES:
        raise ConfigError(
            f"no list price recorded for {model!r}; refusing to report a cost of zero "
            "for a call that was actually billed"
        )
    per_m_in, per_m_out = MODEL_PRICES[model]
    return (input_tokens / 1_000_000) * per_m_in + (output_tokens / 1_000_000) * per_m_out


def cache_key(
    *,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    schema: dict[str, Any] | None,
    thinking: dict[str, Any] | None,
) -> str:
    """Hash every input that could change the output.

    Canonical JSON with sorted keys: otherwise two identical requests whose
    dicts happened to be built in a different order would miss the cache at
    random, which would quietly turn a free re-run into a billed one.
    """
    payload = json.dumps(
        {
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "schema": schema,
            "thinking": thinking,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class Usage:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def as_row(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass(frozen=True)
class LLMResponse:
    parsed: dict[str, Any]
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool
    latency_ms: float

    def require_latency_ms(self) -> float:
        """Latency, or a refusal if it came from the cache."""
        if self.cached:
            raise MeasurementError(
                "this response came from the cache, so its latency is microseconds of "
                "SQL rather than model time; run with cache_bypass to measure latency"
            )
        return self.latency_ms


class Transport(Protocol):
    def __call__(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        schema: dict[str, Any] | None,
        thinking: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


def default_transport(api_key: str | None = None) -> Transport:
    """The real Anthropic transport. Raises if no key is configured.

    The key is only required here, not at import or settings time, so the
    key-free milestones (ingestion and the retrieval tier) keep working.
    """
    if api_key is None:
        from agentforge.settings import settings

        api_key = settings().anthropic_api_key
    if not api_key:
        raise ConfigError(
            "ANTHROPIC_API_KEY is unset. Ingestion and the retrieval tier need no key; "
            "generation and judging do."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    def call(
        *, model, system, messages, max_tokens, schema, thinking
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            # Enforced API-side, so a malformed shape is impossible rather than
            # something to defensively repair.
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": schema}
            }
        if thinking is not None:
            kwargs["thinking"] = thinking

        message = client.messages.create(**kwargs)

        if message.stop_reason == "refusal":
            raise SchemaViolation(
                f"{model} declined the request (stop_reason=refusal, "
                f"details={getattr(message, 'stop_details', None)})"
            )
        if message.stop_reason == "max_tokens":
            raise SchemaViolation(
                f"{model} hit max_tokens={max_tokens} before finishing; the output is "
                "truncated and must not be parsed as if complete"
            )

        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )
        try:
            parsed = json.loads(text) if schema is not None else {"text": text}
        except json.JSONDecodeError as exc:
            raise SchemaViolation(f"{model} returned unparseable JSON: {exc}") from exc

        return {
            "parsed": parsed,
            "input_tokens": message.usage.input_tokens,
            "output_tokens": message.usage.output_tokens,
        }

    return call


def _validate(parsed: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Minimal JSON-Schema check for the subset this project emits.

    Deliberately not a full validator: structured outputs are enforced
    server-side, so this is a second line that catches a drifting schema and
    names the offending field. Raising beats repairing -- a wrong shape is a
    defect.
    """
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(parsed, dict):
            raise SchemaViolation(f"{path}: expected object, got {type(parsed).__name__}")
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in parsed:
                raise SchemaViolation(f"{path}: missing required field {name!r}")
        if schema.get("additionalProperties") is False:
            for name in parsed:
                if name not in properties:
                    raise SchemaViolation(f"{path}: unexpected field {name!r}")
        for name, value in parsed.items():
            if name in properties:
                _validate(value, properties[name], f"{path}.{name}")
    elif expected == "array":
        if not isinstance(parsed, list):
            raise SchemaViolation(f"{path}: expected array, got {type(parsed).__name__}")
        if "items" in schema:
            for index, item in enumerate(parsed):
                _validate(item, schema["items"], f"{path}[{index}]")
    elif expected == "string":
        if not isinstance(parsed, str):
            raise SchemaViolation(f"{path}: expected string, got {type(parsed).__name__}")
        if "enum" in schema and parsed not in schema["enum"]:
            raise SchemaViolation(f"{path}: {parsed!r} is not one of {schema['enum']}")
    elif expected == "boolean":
        if not isinstance(parsed, bool):
            raise SchemaViolation(f"{path}: expected boolean, got {type(parsed).__name__}")
    elif expected == "integer":
        if isinstance(parsed, bool) or not isinstance(parsed, int):
            raise SchemaViolation(f"{path}: expected integer, got {type(parsed).__name__}")
    elif expected == "number":
        if isinstance(parsed, bool) or not isinstance(parsed, (int, float)):
            raise SchemaViolation(f"{path}: expected number, got {type(parsed).__name__}")


@dataclass
class LLMClient:
    model: str
    transport: Transport | None = None
    cache_bypass: bool = False
    max_tokens: int = 8192
    thinking: dict[str, Any] | None = None
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        if self.model not in MODEL_PRICES:
            raise ConfigError(f"no list price recorded for model {self.model!r}")

    def _call(self) -> Transport:
        if self.transport is None:
            self.transport = default_transport()
        return self.transport

    def _cached(self, key: str) -> dict[str, Any] | None:
        with pool().connection() as conn:
            row = conn.execute(
                "SELECT response_json, input_tokens, output_tokens FROM llm_cache "
                "WHERE cache_key = %s",
                (key,),
            ).fetchone()
        if row is None:
            return None
        payload = row["response_json"]
        return {
            "parsed": payload if isinstance(payload, dict) else json.loads(payload),
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
        }

    def _store(self, key: str, result: dict[str, Any]) -> None:
        with pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO llm_cache
                  (cache_key, model, response_json, input_tokens, output_tokens)
                VALUES (%s,%s,%s,%s,%s)
                ON CONFLICT (cache_key) DO NOTHING
                """,
                (
                    key,
                    self.model,
                    json.dumps(result["parsed"]),
                    result["input_tokens"],
                    result["output_tokens"],
                ),
            )

    def complete(
        self,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """One structured completion, cached unless bypassed."""
        messages = [{"role": "user", "content": prompt}]
        limit = max_tokens or self.max_tokens
        key = cache_key(
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=limit,
            schema=schema,
            thinking=self.thinking,
        )

        hit = None if self.cache_bypass else self._cached(key)
        started = time.perf_counter()
        if hit is not None:
            result, cached = hit, True
        else:
            result = self._call()(
                model=self.model,
                system=system,
                messages=messages,
                max_tokens=limit,
                schema=schema,
                thinking=self.thinking,
            )
            cached = False
        elapsed_ms = (time.perf_counter() - started) * 1000

        if schema is not None:
            _validate(result["parsed"], schema)

        if not cached:
            self._store(key, result)

        cost = price(self.model, result["input_tokens"], result["output_tokens"])
        self.usage.calls += 1
        if cached:
            self.usage.cache_hits += 1
        else:
            self.usage.input_tokens += result["input_tokens"]
            self.usage.output_tokens += result["output_tokens"]
            self.usage.cost_usd += cost

        return LLMResponse(
            parsed=result["parsed"],
            model=self.model,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            # A cache hit was not billed, so it is reported as free.
            cost_usd=0.0 if cached else cost,
            cached=cached,
            latency_ms=elapsed_ms,
        )
