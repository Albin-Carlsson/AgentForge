# AgentForge — design decisions

Fifteen decisions, each resolved before any code was written, recorded here with
the reasoning and the rejected alternative. The ordering is the dependency
ordering: each decision constrains the ones below it.

## 0. What this project is for

Résumé proof that survives a 45-minute technical interview. The deliverable is
not the system; it is the ability to defend every decision in it, backed by a
repo someone can clone and run. The failure mode being designed against is not
"incomplete" but "twelve shallow wrappers and no numbers" — because the second
interview question is always *"what did reranking actually buy you?"*

**Consequence:** depth on two subsystems, honest minimal versions of the rest,
and a measured number behind every claim in the résumé bullet.

## 1. Centrepiece: the evaluation harness

Everyone with an LLM project built RAG; almost nobody can state their retrieval
recall@10 or show a regression gate that failed. The eval work is the scarce
skill, and it is what makes the retrieval claims defensible at all — "hybrid
retrieval with reranking" is a claim about quality, so without the harness there
is no baseline to compare against.

**Consequence:** harness before features. Every parameter below is chosen by
measurement rather than by taste, which is also what makes each one an anecdote
with a number attached.

## 2. Topology: a fixed four-role DAG, with a single-agent baseline

`Planner → parallel Researchers → Verifier → Synthesizer`. "Multi-agent" is the
most inflated term in this field, so the justification has to be measurable:

1. **Context isolation.** A single agent accumulates every retrieved chunk from
   every sub-question into one window; by sub-question five, attention is diluted
   by mostly-irrelevant text. Separate Researchers each hold only their own
   evidence.
2. **Adversarial separation.** The Verifier must not see the Synthesizer's
   reasoning — only a claim and its cited span. Sharing context makes a verifier
   rubber-stamp its own prior reasoning, which is what turns evidence
   verification from a real check into decoration.
3. **Parallelism.** Wall-clock latency scales with the slowest sub-question
   rather than their sum.

**The single-agent arm is part of the deliverable, not a straw man.** It converts
"why multi-agent?" from a question of taste into a table.

**Rejected:** a supervisor with dynamic delegation, and agent-to-agent debate.
Both are non-deterministic, expensive, and cannot produce stable numbers.

## 3. Corpus: ~300 arXiv papers on retrieval/RAG, plus linked web sources

The expensive part of evaluation is labels, and writing labels requires
understanding the documents. In a known domain that is an afternoon; in SEC
filings it is a month and the labels are subtly wrong — which is worse than no
labels, because every reported number inherits the error.

Also: multi-hop questions arise naturally ("which of these use BEIR, and how do
their baselines differ?"), so the Planner earns its existence instead of
theatrically splitting a question that did not need splitting.

**Dropped from the original scope:** datasets/CSV analysis. Three input
modalities sound stronger than two, but text-to-SQL is a separate system with a
separate metric (execution accuracy, not groundedness). A modality that folds
under one follow-up question is worth less than two that do not.

## 4. Three evaluation tiers, with distinct jobs

| Tier | Source | N | Labels | Cost | Job |
|---|---|---|---|---|---|
| Retrieval | BEIR SciFact | ~300 queries | gold qrels, free | **$0** | every retrieval ablation; statistical power |
| Product | arXiv, model-drafted + human-reviewed | 40 | gold spans, whole-paper audit | ~$6/arm | the numbers that get quoted |
| Judge validation | hand-labelled pairs | 100 | by hand | ~$1 | κ; licenses the instrument |

Public benchmarks and self-curated sets fail in opposite directions, which is
why both are here. A benchmark gives statistical power (a 2-point delta is real
at N=300; at N=40 the CI is ±7 and reranking's effect is undetectable), free
labels, and external calibration — but it is contaminated by pretraining, ships
as clean pre-chunked text that bypasses the entire ingestion path, and asks for
short factoid answers rather than cited reports. The curated set exercises
the real pipeline end to end, is contamination-free by construction (post-cutoff
papers), and is the only tier that can measure citation precision.

**Dropped:** a 300-question synthetic middle tier. Redundant between the two.

## 5. Grading: claim-decomposed, with the judge itself validated

- **Groundedness and correctness are separate metrics.** A claim can be faithfully
  supported by a retrieved chunk and still be factually wrong. Merging them is
  the most common eval mistake.
- **Atomic claims, binary verdicts.** Not a 1–5 score on a 400-word report, which
  produces noise with a decimal point. Counts are reproducible across runs.
- **κ against reference labels.** ~100 labelled pairs, judge agreement reported.
  Below κ=0.70 the judge is not an instrument and the prompt gets fixed and
  re-measured. An unvalidated judge makes every downstream number decorative.
- **The judge is chosen by measurement, not by price.** Haiku/Sonnet/Opus over the
  same 100 pairs costs about a dollar; the cheapest one clearing 0.70 wins.

## 6. Attribution: config-hashed runs, per-question rows

Eleven knobs affect any reported delta. A run is an immutable row keyed by the
hash of a frozen config; the harness refuses to tabulate a comparison between
configs differing in more than one field (`config.assert_comparable`).

- **Per-question storage, never pre-aggregated.** Aggregates are a `GROUP BY`;
  the per-question rows are what let you diff two runs and find the eight
  questions that regressed. That is where the insight is.
- **Paired comparison.** Both arms always run the identical frozen question set,
  compared with a paired bootstrap. At N=40, pairing is what makes a 5-point
  delta detectable at all.
- **Content-addressed LLM cache.** Re-running is free; a config change re-bills
  only the calls it changed; CI runs offline.
- **Own the eval tables; dual-write scores to Langfuse.** Postgres is already
  present for pgvector, so `runs`/`results` is three tables and gives SQL over
  everything. Langfuse owns per-span traces, token/cost attribution and failure
  drilling. What is avoided is a hosted dependency on the critical path of the
  headline claim.

## 7. Ingestion: PyMuPDF, structure-aware small-to-big

Chunking is not a retrieval detail — the chunk **is** the citation span, so it
sets the resolution of the entire groundedness story. 1024-token chunks make a
citation point at half a page and the judge's entailment call mushy.

- Retrieve and cite at ~256 tokens; hand the generator the expanded parent
  section. Precision for ranking and attribution, context for synthesis.
- Never chunk across a section boundary; carry `section_path` and prefix it plus
  the title into `embed_text` (an arm, not an assumption).
- Every chunk carries `(doc_id, page, char_span, bbox)`, so a citation resolves
  to a highlightable rectangle. "Citation-backed" in the strong sense.
- **The parser is measured too:** section-header recall against ar5iv HTML.
- Chunk size is not chosen, it is tuned: 256/512/1024 as arms on the free tier.

## 8. Lexical retrieval: real BM25 via `pg_search`, fused with RRF

Postgres `ts_rank_cd` is **not** BM25 — different term-frequency saturation,
different length normalisation, no `k1`/`b`. Writing "BM25" on a résumé and
having it turn out to be `tsvector` is the kind of overstatement that costs more
than the keyword gained. ParadeDB's `pg_search` is Tantivy-backed Okapi BM25
inside Postgres, in the same image as pgvector.

**Why one store:** OpenSearch alongside pgvector means a dual-write path and a
window where a chunk is lexically searchable but not semantically — or a reindex
that half-fails. At ~50k chunks that complexity buys nothing. The threshold at
which it would is single-digit millions of chunks.

**RRF, not weighted score addition.** BM25 scores are unbounded and
corpus-dependent; cosine similarity is bounded in a different space. Adding them
needs per-query normalisation that is unstable exactly when one retriever
returns nothing good. RRF consumes only ranks.

**What is actually tunable, verified against the installed engine rather than
the docs.** `pg_search` 0.26.3 does not expose Okapi `k1`/`b`; Tantivy fixes
them at 1.2 and 0.75. An earlier draft of this design listed them as swept
arms, and `RetrievalConfig` deliberately has no field for them -- a config knob
implying a sweep the engine cannot perform is exactly the overstatement
decision 15 forbids. The analyzer *is* configurable, so the real lexical arms
are **stemming** (English/none -- verified to change results: "evaluating"
matches "evaluated" only with the stemmer on) and **stopword removal**. Two
further constraints the code had to be written around: a relation may hold only
one ParadeDB index, so changing the analyzer is a drop-and-recreate (which is
why index building is an explicit step that `search` refuses to perform
implicitly); and `ORDER BY paradedb.score(id)` must be standalone, since
wrapping it silently loses the Top-K index scan.

## 9. No fallbacks, anywhere

Every error raises. No `except: pass`, no degraded path, no defaults for absent
config, no partial ingests. A silent fallback corrupts attribution: a reranker
that quietly no-ops, or an ingest that drops the documents that failed to parse,
inflates the reported numbers.

The boundary — defect versus model-quality outcome — is in
[`src/agentforge/errors.py`](../src/agentforge/errors.py).

## 10. Structured claims with verbatim quotes

The Synthesizer emits `{claim_text, citations: [{chunk_id, verbatim_quote}]}` as
structure; readable prose is rendered from it.

- **Error stays out of the metric path.** Post-hoc decomposition puts an extra
  LLM call inside the measurement: the decomposer's own error rate (mis-split
  compound claims, dropped hedges, misattached citations) is baked into every
  hallucination number, and κ then partly measures the decomposer.
- **Fabrication detection becomes deterministic.** A quote either is or is not an
  exact substring of its chunk. `str.find`, not a model call: zero cost, zero
  variance. This splits the hallucination metric into two mechanically different
  classes — *fabricated quote* (deterministic, should be ~0) and *quote does not
  entail claim* (genuinely fuzzy, where κ matters).
- Normalisation (ligatures, soft hyphens, NBSP from PDF extraction) is applied
  once, canonically, to both sides. It has its own unit tests, because it is the
  seam where a real bug could masquerade as a hallucination.

**Cost accepted:** reports read more staccato than free prose. Right trade for a
research tool; the renderer groups claims by sub-question to recover flow.

## 11. Models

| Role | Choice | Why |
|---|---|---|
| Embeddings | `bge-base-en-v1.5` (440 MB, 768-d) | English-only corpus; MPS makes a full re-embed minutes, so model size is an arm |
| Reranker | `ms-marco-MiniLM-L-6-v2` (90 MB) | tens of ms for 50 candidates on an M3 — the latency story is concrete |
| Generation + judge | `claude-haiku-4-5` | ~$0.15/question; whole project lands ~$20–30 |

Local retrieval models are not a cost decision but a measurement one: a
chunk-size arm requires re-embedding the corpus, and that must be a CPU-minutes
decision rather than an API-bill decision. Anthropic serves no embedding model,
so this was a choice either way.

**Cost as a metric stays honest:** token counts and latency are measured
directly; the dollar column is computed from the published rate card.

## 12. Verifier rejection: defect versus measurement

A model emitting an unsupported claim is not a bug; it is the phenomenon the
system exists to detect. If that crashed the request, the tool could not report
its own primary finding.

- Model-quality outcomes are recorded and **surfaced in the report as explicitly
  unverified**. Dropping them would be a fallback — it hides the failure the
  eval exists to count.
- **One bounded retry** on the verify edge, with the rejection reason fed back to
  the Researcher. Unbounded self-correction is how these graphs become
  unmeasurable. Whether the retry rescues claims or just burns tokens is itself
  an arm.

## 13. Latency and cost, measured honestly

- **The LLM cache destroys latency measurement.** Cached calls return in
  microseconds, so latency comes only from runs flagged `cache_bypass`. Asking
  for latency from a cached run raises `MeasurementError`.
- **p95 is decomposed rather than bought.** Retrieval and rerank p95 come from
  thousands of free queries; end-to-end p95 from 40 questions × 3 cache-bypassed
  repeats (120 observations), run once on the frozen shipped config. Reported
  stage-wise: *end-to-end 6.1 s, of which retrieval 180 ms, rerank 40 ms.*
- **Boundaries stated or the number is meaningless.** Cost per query = serving
  path only (Planner + Researchers + Synthesizer + inline verification),
  excluding the eval-time judge; uncached list prices, cache savings reported
  separately. Latency = warm index, sequential single request, excluding
  one-time ingestion. No concurrency testing — different project.

This is also what earns Langfuse its place: the per-stage spans *are* the
latency decomposition, so it stops being a keyword and becomes where a number
comes from.

## 14. Build order: every milestone ends in a number

| M | Scope | Ends in |
|---|---|---|
| M0 | README + bullet with blanks | the spec |
| M1 | free retrieval tier on SciFact — no LLM, no PDFs, no agents | first ablation table |
| M2 | arXiv ingestion, PyMuPDF, small-to-big | parser section-recall %; M1 arms on the real corpus |
| M3 | single-agent baseline, structured claims, quote-match | fabricated-quote rate |
| M4 | 40 frozen questions, judge, 100 labelled pairs, κ | groundedness + a validated instrument |
| M5 | four-role LangGraph, bounded verify edge | graph vs single-agent |
| M6 | FastAPI, Docker, Langfuse, README, demo | the deliverable |

**M1 is deliberately the riskiest half of the stack** (`pg_search`, hybrid, RRF,
config hashing) so it fails in week one rather than week five. SciFact abstracts
are short, so chunking does not block it.

**LangGraph is M5, not M1** — building orchestration around a retrieval system
whose quality cannot yet be measured is exactly how this becomes twelve shallow
wrappers.

## 15. The deliverable is the README

Four artefacts, and the code serves them: the bullet with numbers in it, a README
an interviewer reads in 90 seconds, a 60-second demo capture, one short write-up
of a surprising finding.

**The governing rule:** every noun in the résumé bullet maps to a file in this
repo and a number in the README. The biggest risk is not an unimpressive project
but one caught overstatement — the moment "datasets" or "BM25" turns out not to
exist as claimed, every other number becomes suspect.


---

## Postscript: where the measurements contradicted this design

Kept deliberately, because a design document that records only its correct
predictions is not evidence of anything.

**Decision 8 predicted hybrid retrieval would win on a technical corpus**, on the
reasoning that papers are dense with rare exact identifiers ("BEIR", "nDCG@10")
where BM25 is strong and dense retrieval weak. Measured on SciFact: hybrid RRF
*does* win on recall@10 (0.8877, best of any arm, +7.5 over BM25) but is
statistically indistinguishable from dense-alone on nDCG@10 (−0.0011, CI
[−0.0223, +0.0202]). The mechanism was half right: fusion improves which
documents are found, not the order they are returned in. That distinction was
only visible because both metrics were reported.

**Decision 11 assumed the cross-encoder would earn its latency.** It did not. On
SciFact the MS MARCO MiniLM costs 4.4 nDCG@10 points (CI [−0.072, −0.016]) and
`bge-reranker-base` still loses recall@10 significantly against no reranking.
Reranking now ships disabled, and the first bullet of the résumé claims hybrid
retrieval rather than reranking as the measured improvement. Keeping the
reranker implemented but off is the honest position: the code exists, the arm is
in the table, and the number says not to use it on this corpus.

**The harness found a bug in itself before it found anything about the system.**
The first ablation scored recall@10 against arms capped at `final_k=8`. Both
errors -- the cap and the mismatch -- were invisible in aggregate: the arms still
produced plausible, internally consistent numbers. What exposed it was the
reranked arm landing within 0.0001 of the BM25 arm, which is not a coincidence
any mechanism explains. `harness.per_query_metric` now raises
`MeasurementError` when asked for a cutoff deeper than the run's `final_k`.


**Decision 7 said the parser would be measured. Measuring it changed what the
measurement was.** Three times, the first number out of the fidelity harness
described the harness rather than the parser:

1. ar5iv was the obvious ground truth and the wrong one -- it has no rendering
   for recent papers and *redirects to the abstract page* instead of returning
   404, so every paper scored as unmeasurable. arXiv's own LaTeXML rendering
   works; the absence of section-title classes is now what detects a missing
   rendering.
2. "Abstract", "Keywords" and "References" scored as spurious parser output on
   every paper, because LaTeXML gives them their own classes rather than
   `ltx_title_section`. They are real sections.
3. Roman numerals without a trailing period went unstripped, so one paper the
   parser had read correctly scored recall 0.025 -- "I Introduction" against
   "I. INTRODUCTION".

And one judgement call worth defending rather than hiding: run-in
`\paragraph{}` labels are excluded from the recall denominator. They are real
structure in the author's source, but they render inline with their prose, so a
line-based PDF parser cannot detect them by construction. On arXiv:2610.32318
they are 25 of 38 LaTeXML headings and alone pushed apparent recall to 0.158.
Excluding them is defining the metric to match the claim being made -- that the
parser recovers section structure, not that it recovers every run-in label.

### Reranking changed sign with the domain (§8, §11)

SciFact said reranking loses, so it shipped off with a note to re-measure on
the arXiv tier. Re-measured, `bge-reranker-base` over plain hybrid gains +0.181
nDCG@10 [+0.103, +0.261], so it ships on. The SciFact result was not wrong. It
answered a question about a different corpus, which is the argument for the
served corpus having its own tier.

### Models: Groq, not Anthropic (§11)

The only key available was a Groq free tier, so generation runs on
`openai/gpt-oss-120b` and judging on `openai/gpt-oss-20b`. That keeps the judge
a different model from the generator, and gives it a separate daily quota. The
quota (200K tokens/day per model) is smaller than one pass of the four-role
arm, which made the content-addressed cache load-bearing: a run stopped by the
daily quota resumes the next day by replaying completed calls for free. The
contamination argument still holds empirically. The no-retrieval control
scores 0.021 correctness on the post-cutoff corpus.

### No Synthesizer model, no web sources (§2, §3)

The four-role graph never got an LLM Synthesizer. The report is rendered
deterministically from the verified claims, which keeps one more model call out
of the measurement path. That fits §2's own argument, so it stays. Web-source
ingestion was not built, and the README no longer claims it.

### Correctness is deterministic, not judged (§5)

A correctness judge would have been a second instrument needing its own κ.
Each frozen question instead carries `answer_keys`, the facts a correct answer
must state, scored by normalised string match. It is strict ("seventy-six
percent" misses "76%"), and the strictness is identical across arms, so the
deltas are unaffected.

### Two measurement artefacts found while building the reference pairs

`canonical()` joins hyphen line breaks, which also deletes the hyphen of a
genuine compound split across lines. A model quoting "English-to-Vietnamese"
exactly was scored as fabricating. Quote matching is now hyphen-insensitive.
And the test suite shared the real database while its cleanup deleted
`llm_cache`, so every test run silently discarded the cache a multi-day run
resumes from. Tests now run against a derived `<db>_test` database.
