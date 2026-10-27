-- AgentForge schema.
--
-- Design notes that matter for reading this file:
--
--  * pg_search and pgvector both live here, so the lexical and dense indexes are
--    the same rows in the same transaction. There is no sync path between stores.
--  * Every measured artefact is keyed by run_id, and every run is keyed by a
--    config_hash covering all eleven knobs plus the git SHA. Two runs are only
--    comparable when their configs differ in exactly one field; the harness
--    asserts that rather than trusting the operator to remember.
--  * Results are stored per question, never pre-aggregated. Aggregates are a
--    GROUP BY; the per-question rows are what let you diff two runs and find the
--    handful of questions that regressed.
--  * Embeddings do NOT live in `chunks`. Embedding dimension is an experiment
--    arm (384 / 768 / 1024), and a pgvector column is fixed-dimension, so each
--    embedding model gets its own table created by retrieval.embed.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

-- ---------------------------------------------------------------- corpus side

CREATE TABLE IF NOT EXISTS corpora (
    corpus_id     text PRIMARY KEY,          -- 'scifact', 'arxiv-rag-300'
    kind          text NOT NULL,             -- 'beir' | 'arxiv' | 'web'
    ingest_config jsonb NOT NULL,            -- parser + chunker params, hashed into run configs
    n_documents   integer NOT NULL DEFAULT 0,
    n_chunks      integer NOT NULL DEFAULT 0,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id      text PRIMARY KEY,
    corpus_id   text NOT NULL REFERENCES corpora(corpus_id) ON DELETE CASCADE,
    source_type text NOT NULL,               -- 'pdf' | 'web' | 'beir'
    source_ref  text NOT NULL,               -- file path, URL, or dataset doc id
    title       text,
    published   date,                        -- used to prove post-cutoff, contamination-free
    meta        jsonb NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS documents_corpus_idx ON documents (corpus_id);

-- A chunk is both the retrieval unit and the citation unit: its span is what a
-- claim's verbatim quote must be found inside. `id` is a surrogate because
-- pg_search wants an integer key field.
CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial PRIMARY KEY,
    chunk_id     text UNIQUE NOT NULL,
    doc_id       text NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    corpus_id    text NOT NULL REFERENCES corpora(corpus_id) ON DELETE CASCADE,
    ordinal      integer NOT NULL,           -- position within the document
    section_path text,                       -- '4. Experiments > 4.2 Ablations'
    page         integer,                    -- 1-indexed, PDFs only
    char_start   integer,
    char_end     integer,
    bbox         real[],                      -- [x0,y0,x1,y1] on `page`, PDFs only
    dom_path     text,                        -- web sources, the analogue of page+bbox
    text         text NOT NULL,               -- verbatim source text; quotes match against THIS
    embed_text   text NOT NULL,               -- text + contextual prefix; what gets embedded
    parent_text  text,                        -- expanded parent section, for small-to-big
    n_tokens     integer
);

CREATE INDEX IF NOT EXISTS chunks_corpus_idx ON chunks (corpus_id);
CREATE INDEX IF NOT EXISTS chunks_doc_idx    ON chunks (doc_id, ordinal);

-- ------------------------------------------------------------ evaluation side

-- Tiers: 'scifact'  = free retrieval tier, gold qrels, no LLM, large N.
--        'arxiv40'  = curated (data/arxiv40.jsonl), frozen, gold spans, contamination-free.
CREATE TABLE IF NOT EXISTS queries (
    query_id    text PRIMARY KEY,
    tier        text NOT NULL,
    corpus_id   text NOT NULL REFERENCES corpora(corpus_id) ON DELETE CASCADE,
    text        text NOT NULL,
    gold_answer text,                         -- arxiv40 only
    frozen_at   timestamptz NOT NULL DEFAULT now(),
    meta        jsonb NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS queries_tier_idx ON queries (tier);

-- Relevance judgements. doc_id for BEIR (document-level qrels), chunk_id for the
-- curated arxiv tier (span-level gold).
CREATE TABLE IF NOT EXISTS qrels (
    query_id  text NOT NULL REFERENCES queries(query_id) ON DELETE CASCADE,
    doc_id    text,
    chunk_id  text,
    relevance integer NOT NULL,
    CHECK (doc_id IS NOT NULL OR chunk_id IS NOT NULL)
);

CREATE INDEX IF NOT EXISTS qrels_query_idx ON qrels (query_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id       text PRIMARY KEY,
    config_hash  text NOT NULL,
    config_json  jsonb NOT NULL,
    git_sha      text NOT NULL,
    arm_label    text NOT NULL,               -- 'hybrid+rerank', 'single-agent', ...
    tier         text NOT NULL,
    cache_bypass boolean NOT NULL DEFAULT false,  -- latency is only honest when true
    started_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz,
    notes        text
);

CREATE INDEX IF NOT EXISTS runs_config_idx ON runs (config_hash);
CREATE INDEX IF NOT EXISTS runs_tier_idx   ON runs (tier, arm_label);

-- Per-question retrieval output: the ranked list, as retrieved. Retrieval
-- metrics (recall@k, nDCG@10, MRR) are computed from these rows and need no LLM,
-- which is what makes the whole retrieval ablation free.
CREATE TABLE IF NOT EXISTS retrieval_results (
    run_id    text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    query_id  text NOT NULL REFERENCES queries(query_id) ON DELETE CASCADE,
    stage     text NOT NULL,                  -- 'bm25' | 'dense' | 'fused' | 'reranked'
    rank      integer NOT NULL,
    chunk_id  text NOT NULL,
    doc_id    text NOT NULL,
    score     double precision NOT NULL,
    PRIMARY KEY (run_id, query_id, stage, rank)
);

-- Per-question end-to-end output.
CREATE TABLE IF NOT EXISTS answers (
    run_id            text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    query_id          text NOT NULL REFERENCES queries(query_id) ON DELETE CASCADE,
    answer_md         text NOT NULL,
    stage_latency_ms  jsonb NOT NULL,         -- {'retrieve':180,'rerank':40,'synthesize':4100}
    latency_ms        integer NOT NULL,
    input_tokens      integer NOT NULL,
    output_tokens     integer NOT NULL,
    cached_tokens     integer NOT NULL DEFAULT 0,
    cost_usd          numeric(12,6) NOT NULL, -- measured tokens priced at list
    PRIMARY KEY (run_id, query_id)
);

-- One row per atomic claim. The Synthesizer emits these as structure, so this is
-- not a post-hoc decomposition -- there is no decomposer whose error rate would
-- leak into the hallucination metric.
--
-- quote_found is a deterministic exact substring check against chunks.text after
-- canonical normalisation: no model call, no variance. entailed is the LLM judge.
CREATE TABLE IF NOT EXISTS claims (
    run_id      text NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    query_id    text NOT NULL REFERENCES queries(query_id) ON DELETE CASCADE,
    claim_idx   integer NOT NULL,
    claim_text  text NOT NULL,
    chunk_id    text NOT NULL,
    quote       text NOT NULL,
    quote_found boolean NOT NULL,             -- deterministic; false == fabricated quote
    entailed    text,                         -- 'entailed'|'contradicted'|'unsupported'
    judge_model text,
    verify_pass integer NOT NULL DEFAULT 0,   -- 0 = first attempt, 1 = after the one retry
    PRIMARY KEY (run_id, query_id, claim_idx)
);

-- Human (or pending-review) reference labels for judge validation. Cohen's kappa
-- is computed between `label` here and the judge's verdict on the same pair.
CREATE TABLE IF NOT EXISTS judge_reference (
    pair_id      text PRIMARY KEY,
    claim_text   text NOT NULL,
    quote        text NOT NULL,
    chunk_id     text NOT NULL,
    label        text NOT NULL,               -- 'entailed'|'contradicted'|'unsupported'
    label_source text NOT NULL,               -- 'model-pending-review' | 'human-reviewed'
    reviewed_at  timestamptz
);

CREATE TABLE IF NOT EXISTS judge_validation (
    judge_model  text NOT NULL,
    prompt_ver   text NOT NULL,
    n_pairs      integer NOT NULL,
    kappa        double precision NOT NULL,
    agreement    double precision NOT NULL,
    label_source text NOT NULL,
    measured_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (judge_model, prompt_ver, label_source)
);

-- Content-addressed LLM cache: re-running an experiment is free, and a config
-- change only re-bills the calls it actually changed.
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key     text PRIMARY KEY,           -- sha256(model, params, prompt)
    model         text NOT NULL,
    response_json jsonb NOT NULL,
    input_tokens  integer NOT NULL,
    output_tokens integer NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- Research requests. A run takes tens of seconds, so the API hands back a job
-- id immediately and an in-process asyncio task writes progress here. No
-- Celery, no Redis: one table and one task is the whole queue, which is the
-- right size for this system.
CREATE TABLE IF NOT EXISTS research_jobs (
    job_id       text PRIMARY KEY,
    question     text NOT NULL,
    arm          text NOT NULL,              -- 'single_agent' | 'four_role'
    status       text NOT NULL,              -- 'queued'|'running'|'done'|'failed'
    report_md    text,
    claims_json  jsonb,
    counts_json  jsonb,
    stage_latency_ms jsonb,
    usage_json   jsonb,
    error        text,                       -- failures are recorded, not swallowed
    created_at   timestamptz NOT NULL DEFAULT now(),
    finished_at  timestamptz
);

CREATE INDEX IF NOT EXISTS research_jobs_status_idx ON research_jobs (status, created_at);
