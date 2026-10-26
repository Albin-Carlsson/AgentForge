# PDF parser fidelity

Ground truth is arXiv's own LaTeXML rendering (`arxiv.org/html/<id>`), so
the reference sectioning comes from the authors' LaTeX rather than from how
the PDF happens to look. No labelling, no cost.

**Read the two metrics separately.** A missed heading merges its body into the
previous section: citations get coarser, but every character is still chunked
and retrievable. Uncovered text is the failure that would actually invalidate a
retrieval number. Coverage below 1.0 is expected -- title, authors and
affiliations precede the first heading and belong to no section.

Run-in `\paragraph{}` labels are counted but NOT scored: they render inline
with their prose, so a line-based parser cannot see them. On one paper they are
25 of 38 LaTeXML headings, and including them would have reported the metric's
definition rather than the parser's quality.

| Paper | pages | text coverage | section recall | section precision | truth | found | run-in (unscored) | note |
|---|---|---|---|---|---|---|---|---|
| 2610.29464v1 | 13 | 0.993 | 0.179 | 0.875 | 39 | 8 | — |  |
| 2610.30612v1 | 18 | 0.961 | 0.786 | 0.917 | 42 | 37 | — |  |
| 2610.30697v1 | 14 | 0.995 | 0.318 | 0.700 | 22 | 10 | 10 |  |
| 2610.31959v1 | 23 | 0.964 | 0.946 | 0.946 | 37 | 37 | 2 |  |
| 2610.32318v1 | 8 | 0.965 | 0.462 | 0.857 | 13 | 7 | 25 |  |
| 2610.32608v1 | 14 | 0.990 | 0.292 | 0.636 | 24 | 11 | — |  |

- Papers measured: 6; with ground truth: 6
- Mean text coverage: **0.978**
- Mean section recall: **0.497**
- Mean section precision: **0.822**
