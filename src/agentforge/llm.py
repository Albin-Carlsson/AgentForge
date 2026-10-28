"""LLM client (Anthropic or Groq) with a content-addressed cache and honest accounting.

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

Structured output is enforced API-side -- `output_config.format` on Anthropic,
strict `json_schema` constrained decoding on Groq -- and validated again here. A schema violation raises: there is no repair path, because a model
that returned the wrong shape is a defect, not a measurement.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from agentforge.config import MODEL_PRICES, MODEL_PROVIDERS
from agentforge.db.conn import pool
from agentforge.errors import ConfigError, MeasurementError, QuotaExhausted, SchemaViolation


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
        output_config: dict[str, Any] = {}
        if schema is not None:
            # Enforced API-side, so a malformed shape is impossible rather than
            # something to defensively repair.
            output_config["format"] = {"type": "json_schema", "schema": schema}
        thinking = dict(thinking or {})
        if "effort" in thinking:
            # Effort lives in output_config on current models, not in `thinking`.
            output_config["effort"] = thinking.pop("effort")
        if output_config:
            kwargs["output_config"] = output_config
        if thinking:
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


GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# A per-minute limit resets within a minute and is worth waiting for. A longer
# retry-after means the daily quota is spent: stop, and resume from the cache.
MAX_RATE_LIMIT_WAIT_S = 90.0
MAX_RATE_LIMIT_RETRIES = 6
# A dropped connection or DNS blip is resent unchanged, a bounded number of
# times. The same request either succeeds or raises; nothing is degraded.
MAX_NETWORK_RETRIES = 3
# Groq answers 400 json_validate_failed when a sampled generation did not
# complete under strict decoding -- observed with an empty failed_generation,
# i.e. no output at all. The identical request is resent at most this many
# times, each one logged. A resend cannot select on the verdict: the failed
# attempt produced none.
MAX_INVALID_GENERATION_RETRIES = 2


def groq_transport(api_key: str | None = None, sleep=time.sleep, post=None) -> Transport:
    """The Groq transport: OpenAI-compatible chat completions, strict schemas.

    `post` is injectable so the rate-limit handling can be tested without
    spending quota; it takes (url, headers, body) and returns
    (status, headers, body).
    """
    if api_key is None:
        from agentforge.settings import settings

        api_key = settings().groq_api_key
    if not api_key:
        raise ConfigError("GROQ_API_KEY is unset; Groq-served models need it.")

    import ssl

    import certifi

    # The python.org macOS build ships without a CA bundle, so the default
    # context cannot verify anything. certifi's bundle, never verification off.
    tls = ssl.create_default_context(cafile=certifi.where())

    def _post(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict, bytes]:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=180, context=tls) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers), exc.read()

    raw_send = post or _post

    def send(url: str, headers: dict[str, str], body: bytes) -> tuple[int, dict, bytes]:
        for attempt in range(MAX_NETWORK_RETRIES + 1):
            try:
                return raw_send(url, headers, body)
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == MAX_NETWORK_RETRIES:
                    raise
                sleep(5.0 * (attempt + 1))
        raise AssertionError("unreachable")

    def call(*, model, system, messages, max_tokens, schema, thinking) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "max_completion_tokens": max_tokens,
            # Reasoning text is billed either way; not returning it saves bytes
            # and keeps it out of the cache, where it is never read.
            "include_reasoning": False,
        }
        if thinking is not None:
            payload.update(thinking)
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "strict": True, "schema": schema},
            }
        body = json.dumps(payload).encode()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Cloudflare in front of Groq rejects urllib's default agent.
            "User-Agent": "agentforge/0.1",
        }

        invalid_generations = 0
        for _ in range(MAX_RATE_LIMIT_RETRIES + MAX_INVALID_GENERATION_RETRIES + 1):
            status, response_headers, raw = send(GROQ_URL, headers, body)
            if (
                status == 400
                and b"json_validate_failed" in raw
                and invalid_generations < MAX_INVALID_GENERATION_RETRIES
            ):
                invalid_generations += 1
                print(
                    f"warning: {model} returned json_validate_failed; resending "
                    f"({invalid_generations}/{MAX_INVALID_GENERATION_RETRIES})",
                    file=sys.stderr, flush=True,
                )
                continue
            if status != 429:
                break
            lowered = {k.lower(): v for k, v in response_headers.items()}
            wait = float(lowered.get("retry-after", "inf"))
            if wait > MAX_RATE_LIMIT_WAIT_S:
                raise QuotaExhausted(
                    f"{model}: rate limited with retry-after={wait:.0f}s, which is the "
                    "daily quota rather than the per-minute one. Re-run after the reset; "
                    "completed calls replay from the cache at no cost."
                )
            sleep(wait + 0.5)
        else:
            raise QuotaExhausted(
                f"{model}: still rate limited after {MAX_RATE_LIMIT_RETRIES} waits"
            )

        if status != 200:
            raise SchemaViolation(f"{model}: HTTP {status}: {raw[:500].decode(errors='replace')}")
        data = json.loads(raw)
        choice = data["choices"][0]
        if choice["finish_reason"] == "length":
            raise SchemaViolation(
                f"{model} hit max_tokens={max_tokens} before finishing (reasoning tokens "
                "count toward it); the output is truncated and must not be parsed"
            )
        if choice["finish_reason"] != "stop":
            raise SchemaViolation(f"{model}: finish_reason={choice['finish_reason']!r}")
        text = choice["message"]["content"] or ""
        try:
            parsed = json.loads(text) if schema is not None else {"text": text}
        except json.JSONDecodeError as exc:
            raise SchemaViolation(f"{model} returned unparseable JSON: {exc}") from exc
        return {
            "parsed": parsed,
            "input_tokens": data["usage"]["prompt_tokens"],
            "output_tokens": data["usage"]["completion_tokens"],
        }

    return call


def transport_for(model: str) -> Transport:
    provider = MODEL_PROVIDERS.get(model)
    if provider == "anthropic":
        return default_transport()
    if provider == "groq":
        return groq_transport()
    raise ConfigError(f"no provider recorded for model {model!r}")


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
    # Every call is reported, cache hits included and marked as such, so a trace
    # can never be mistaken for a record of model time.
    observer: Any = None
    stage: str = "llm"

    def __post_init__(self) -> None:
        if self.model not in MODEL_PRICES:
            raise ConfigError(f"no list price recorded for model {self.model!r}")

    def _call(self) -> Transport:
        if self.transport is None:
            self.transport = transport_for(self.model)
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

        response = LLMResponse(
            parsed=result["parsed"],
            model=self.model,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            # A cache hit was not billed, so it is reported as free.
            cost_usd=0.0 if cached else cost,
            cached=cached,
            latency_ms=elapsed_ms,
        )
        if self.observer is not None:
            self.observer.record_generation(
                model=self.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost_usd=response.cost_usd,
                cached=response.cached,
                latency_ms=response.latency_ms,
                stage=self.stage,
            )
        return response


def _thinking_for(model: str, reasoning_effort: str | None) -> dict[str, Any] | None:
    """The effort setting in each provider's own shape. Part of the cache key."""
    if reasoning_effort is None:
        return None
    provider = MODEL_PROVIDERS.get(model)
    if provider == "groq":
        return {"reasoning_effort": reasoning_effort}
    if provider == "anthropic":
        return {"effort": reasoning_effort}
    raise ConfigError(f"no provider recorded for model {model!r}")


def generator_client(generation: Any, **kwargs: Any) -> LLMClient:
    """The generator, built from a GenerationConfig so the run hash describes it."""
    return LLMClient(
        model=generation.model,
        max_tokens=generation.max_tokens,
        thinking=_thinking_for(generation.model, generation.reasoning_effort),
        stage="generate",
        **kwargs,
    )


# A verdict is one enum word; the budget is for the reasoning in front of it.
JUDGE_MAX_TOKENS = 768


def judge_client(judge: Any, **kwargs: Any) -> LLMClient:
    """The entailment judge, built from a JudgeConfig."""
    return LLMClient(
        model=judge.model,
        max_tokens=JUDGE_MAX_TOKENS,
        thinking=_thinking_for(judge.model, judge.reasoning_effort),
        stage="judge",
        **kwargs,
    )
