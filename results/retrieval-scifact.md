# Retrieval ablation: scifact

- Engine: PostgreSQL 18.6 (Debian 18.6-1.pgdg13+2), pg_search 0.26.3, pgvector 0.8.6
- BM25 constants: k1=1.2, b=0.75 (tantivy-default, tunable: no)
- Embedding/rerank device: mps
- Scoring cutoff k=10, pipeline final_k=20 (final_k must be >= k or every arm is capped at recall@final_k)

| Arm                               |   recall@10 |   nDCG@10 |    MRR |   n | knob changed           | paired Δ ndcg@10 (95% CI)     |
|-----------------------------------|-------------|-----------|--------|-----|------------------------|-------------------------------|
| BM25 only (pg_search)             |      0.8127 |    0.6843 | 0.6519 | 300 | baseline           | n/a                             |
| Dense only (pgvector)             |      0.8676 |    0.74   | 0.7079 | 300 | retrieval.retriever    | +0.0556 [+0.0227, +0.0880]    |
| Hybrid (RRF)                      |      0.8877 |    0.7389 | 0.7008 | 300 | retrieval.retriever    | -0.0011 [-0.0223, +0.0202] ns |
| Hybrid + MiniLM rerank (MS MARCO) |      0.8272 |    0.6952 | 0.6665 | 300 | retrieval.rerank_model | -0.0436 [-0.0716, -0.0157]    |
| Hybrid + bge-reranker-base        |      0.8488 |    0.722  | 0.6927 | 300 | retrieval.rerank_model | +0.0268 [-0.0006, +0.0546] ns |

Each row differs from the row above by exactly one knob; `assert_comparable`
refuses any comparison that moves two. `ns` marks a delta whose 95% paired
bootstrap interval includes zero -- i.e. not distinguishable from noise at
n=300.
