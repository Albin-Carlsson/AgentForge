# End-to-end — 40 arXiv questions

Generator `claude-haiku-5-5` (effort=low), judge `claude-haiku-5-5`, retrieval: hybrid, final_k=8.
Correctness is deterministic key-fact recall against each question's
`answer_keys`; groundedness is verified citations over all citations.

| Arm | correctness | citations | groundedness | fabricated quotes | unsupported (of found) | cited doc in gold | gold chunk cited | tokens/question | list cost |
|---|---|---|---|---|---|---|---|---|---|
| No-retrieval control | **0.067** | 0 | — | — | — | — | — | 502 | $0.0051 |
| Single agent | **0.829** | 170 | 0.535 | 0.018 | 0.455 | 0.988 | 0.925 | 6,684 | $0.0374 |
| Four-role graph | **0.873** | 462 | 0.630 | 0.000 | 0.370 | 0.952 | 0.975 | 20,466 | $0.1131 |
| Four-role graph, no retry | **0.873** | 427 | 0.614 | 0.033 | 0.366 | 0.948 | 0.975 | 18,175 | $0.1010 |

Paired bootstrap on correctness (10,000 resamples, 95% CI):

- No-retrieval control -> Single agent: correctness improved by +0.7625 [95% CI +0.6375, +0.8750], n=40
- Single agent -> Four-role graph: correctness improved by +0.0437 [95% CI -0.0500, +0.1479], n=40 (not significant)
- Four-role graph, no retry -> Four-role graph: correctness improved by +0.0000 [95% CI +0.0000, +0.0000], n=40 (not significant)

Paired bootstrap on per-question groundedness (questions with citations in both):

- Single agent -> Four-role graph: groundedness improved by +0.0763 [95% CI -0.0191, +0.1733], n=38 (not significant)
- Four-role graph, no retry -> Four-role graph: groundedness improved by +0.0127 [95% CI -0.0027, +0.0296], n=40 (not significant)

The bounded retry fired on 8 of 40 questions. Fabricated quotes: 14 without it, 0 with it.

Runs: `arxiv40-control-edb32afe7d7d-7bc89a`, `arxiv40-single-90b922a59fe7-897414`, `arxiv40-graph-4309dd4cf6b4-cc87b9`, `arxiv40-graph-noretry-8db6daa7ae70-ed4e63`
