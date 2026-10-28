# Retrieval ablation — arxiv40

- Engine: PostgreSQL 18.6 (Debian 18.6-1.pgdg13+2), pg_search 0.26.3, pgvector 0.8.6
- BM25 constants: k1=1.2, b=0.75 (tantivy-default, tunable: no)
- Embedding/rerank device: mps
- Scoring cutoff k=10, pipeline final_k=20 (final_k must be >= k or every arm is capped at recall@final_k)

| Arm                               |   recall@10 |   nDCG@10 |    MRR |   n | knob changed           | paired Δ ndcg@10 (95% CI)     |
|-----------------------------------|-------------|-----------|--------|-----|------------------------|-------------------------------|
| BM25 only (pg_search)             |      0.6441 |    0.5027 | 0.5724 |  40 | — (baseline)           | —                             |
| Dense only (pgvector)             |      0.5404 |    0.4156 | 0.4938 |  40 | retrieval.retriever    | -0.0871 [-0.2270, +0.0488] ns |
| Hybrid (RRF)                      |      0.6648 |    0.5017 | 0.5556 |  40 | retrieval.retriever    | +0.0861 [+0.0206, +0.1511]    |
| Hybrid + MiniLM rerank (MS MARCO) |      0.6864 |    0.5809 | 0.7001 |  40 | retrieval.rerank_model | +0.0792 [-0.0053, +0.1687] ns |
| Hybrid + bge-reranker-base        |      0.7497 |    0.6824 | 0.8027 |  40 | retrieval.rerank_model | +0.1015 [+0.0385, +0.1746]    |

Each row differs from the row above by exactly one knob; `assert_comparable`
refuses any comparison that moves two. `ns` marks a delta whose 95% paired
bootstrap interval includes zero -- i.e. not distinguishable from noise at
n=40.
