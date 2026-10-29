"""Frozen run configuration and its content hash.

The reason this module exists at all: "hybrid retrieval with reranking improved
nDCG@10 by 6 points" is only true if nothing else moved between the two runs.
Eleven knobs affect that number. So a run is an immutable config object, its
hash is recorded alongside every result row, and the harness refuses to tabulate
a comparison between two runs whose configs differ in more than one field.

What is deliberately NOT in the hash:

  git_sha       Recorded on the run, but excluded -- otherwise every commit would
                make two runs differ in two fields and no comparison would ever
                be legal. The comparison report annotates SHA mismatches instead.
  cache_bypass  Affects timing only, never outputs. Recorded on the run because
                latency numbers are only honest when it is true.
  arm_label     A human name for the run. Cosmetic.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentforge.errors import ConfigError, IncomparableRuns

Retriever = Literal["bm25", "dense", "hybrid"]
Graph = Literal["none", "single_agent", "four_role"]

# Embedding models are an arm on the free retrieval tier: a full re-embed costs
# only MPS minutes, so model size becomes a measurement rather than a default.
EMBED_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-m3": 1024,
}

# Measured token counts are priced at list. USD per 1M tokens, (input, output).
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    # Prompts up to 100K tokens; this project's never exceed ~4K.
    "claude-haiku-5-5": (0.10, 0.50),
}




class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IngestConfig(_Frozen):
    """Parsing and chunking. Changing any of these invalidates the index."""

    corpus_id: str
    parser_version: str = "pymupdf-1"
    chunk_tokens: int = 256          # retrieval + citation granularity
    chunk_overlap: int = 38          # ~15%
    parent_section: bool = True      # small-to-big: cite the chunk, generate from the section
    contextual_prefix: bool = True   # title + section_path prepended to embed_text


class RetrievalConfig(_Frozen):
    """Everything between a query string and a ranked list of chunks."""

    embed_model: str = "BAAI/bge-base-en-v1.5"
    retriever: Retriever = "hybrid"
    # Okapi k1/b are deliberately absent: pg_search 0.26 does not expose them
    # (Tantivy fixes k1=1.2, b=0.75), and a config field implying a sweep the
    # engine cannot perform is the overstatement this project refuses to ship.
    # What IS tunable on the analyzer, and is therefore an arm:
    bm25_stemmer: str | None = "English"
    bm25_stopwords: str | None = "English"
    candidate_k: int = 50            # per retriever, before fusion
    rrf_k: int = 60                  # RRF: rank-based, so BM25 and cosine never get added
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    # Measured ON for the product corpus. On BEIR SciFact (n=300) reranking lost:
    # the MS MARCO MiniLM cost 4.4 nDCG@10 points (95% CI [-0.072, -0.016]) and
    # bge-reranker-base lost recall@10. Re-measured on the arXiv tier (n=40), the
    # corpus this system actually serves, bge-reranker-base over plain hybrid
    # gains +0.181 nDCG@10 [+0.103, +0.261] and +0.085 recall@10 [+0.024, +0.158].
    # The domain reversed the sign, which is why it was re-measured rather than
    # carried over. See results/retrieval-{scifact,arxiv40}.md.
    rerank_model: str | None = "BAAI/bge-reranker-base"
    rerank_top_k: int = 50           # candidates fed to the cross-encoder
    final_k: int = 8                 # chunks handed to the generator

    @property
    def embed_dim(self) -> int:
        if self.embed_model not in EMBED_DIMS:
            raise ConfigError(
                f"unknown embed_model {self.embed_model!r}; "
                f"add its dimension to EMBED_DIMS"
            )
        return EMBED_DIMS[self.embed_model]


class GenerationConfig(_Frozen):
    """The agent graph and the model driving it."""

    graph: Graph = "four_role"
    # The cheapest current Claude model.
    model: str = "claude-haiku-5-5"
    max_tokens: int = 3072
    # Claude's output_config.effort. "low" is the cheapest; it is part of every
    # cache key and run hash.
    effort: str | None = "low"
    prompt_version: str = "v1"
    n_subquestions: int = 4              # Planner fan-out width
    max_verify_retries: int = 1          # bounded: one retry, then the claim ships flagged

    @model_validator(mode="after")
    def _check(self) -> GenerationConfig:
        if self.model not in MODEL_PRICES:
            raise ConfigError(f"no list price recorded for model {self.model!r}")
        if self.max_verify_retries > 1:
            raise ConfigError(
                "max_verify_retries > 1: unbounded self-correction makes the graph "
                "unpredictable and the latency distribution unmeasurable"
            )
        return self


class JudgeConfig(_Frozen):
    """The measuring instrument. Frozen once validated, and versioned."""

    # The same model as the generator: the owner set a Claude-only, cheapest-
    # model budget, and a stronger Claude judge would cost most of it. The
    # separation that remains is context: the judge sees one claim and one
    # quote, never the question or the generator's reasoning. Validated by
    # `validate-judge`; see results/judge-validation.md.
    model: str = "claude-haiku-5-5"
    effort: str | None = "low"
    prompt_version: str = "v1"
    min_kappa: float = 0.70              # below this the judge is not a usable instrument


class RunConfig(_Frozen):
    """The full experiment definition. Hashable, comparable, serialisable."""

    ingest: IngestConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

    def canonical(self) -> str:
        """Deterministic JSON: sorted keys, no whitespace. The hash input."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:12]

    def flat(self) -> dict[str, object]:
        """Flatten to dotted paths, so a diff names the exact knob that moved."""
        out: dict[str, object] = {}
        for section, values in self.model_dump(mode="json").items():
            for key, value in values.items():
                out[f"{section}.{key}"] = value
        return out

    def diff(self, other: RunConfig) -> dict[str, tuple[object, object]]:
        mine, theirs = self.flat(), other.flat()
        return {
            key: (mine[key], theirs[key])
            for key in sorted(mine.keys() | theirs.keys())
            if mine.get(key) != theirs.get(key)
        }


def assert_comparable(a: RunConfig, b: RunConfig) -> str:
    """Return the single knob that differs, or refuse.

    This is the whole point of the config hash. A two-field difference means the
    delta cannot be attributed, so the harness will not produce the table.
    """
    delta = a.diff(b)
    if len(delta) == 1:
        return next(iter(delta))
    if not delta:
        raise IncomparableRuns(
            f"configs are identical ({a.config_hash}): there is no knob to attribute a delta to"
        )
    raise IncomparableRuns(
        "cannot attribute a delta between configs differing in "
        f"{len(delta)} fields: " + ", ".join(f"{k} {v[0]!r}->{v[1]!r}" for k, v in delta.items())
    )


def git_sha() -> str:
    """Recorded on every run. Not hashed -- see the module docstring."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise ConfigError(
            "cannot read git SHA; every run records the code revision that produced it"
        ) from exc
    return result.stdout.strip()
