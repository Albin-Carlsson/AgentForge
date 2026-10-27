"""Command line entry points. Thin: every command delegates and prints.

Deliberately has no `--force`, no `--skip-errors` and no `--continue-on-failure`
flag. A partial ingest or a half-built index silently changes what every
reported number means.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from agentforge.config import IngestConfig, RetrievalConfig, RunConfig
from agentforge.db.conn import apply_schema, close_pool, server_info
from agentforge.eval import harness, report
from agentforge.retrieval import lexical
from agentforge.retrieval.embed import Embedder

RESULTS = pathlib.Path("results")

# The M1 ladder: four arms, each one knob from its predecessor.
LADDER = [
    ("BM25 only (pg_search)", {"retriever": "bm25", "rerank_model": None}),
    ("Dense only (pgvector)", {"retriever": "dense", "rerank_model": None}),
    ("Hybrid (RRF)", {"retriever": "hybrid", "rerank_model": None}),
    ("Hybrid + MiniLM rerank (MS MARCO)", {"retriever": "hybrid"}),
    # Fifth rung, one knob from the fourth. The MiniLM cross-encoder is trained
    # on MS MARCO web-search pairs while SciFact is scientific claim
    # verification, so "reranking hurts" and "this reranker hurts" are different
    # claims and the ladder should distinguish them.
    (
        "Hybrid + bge-reranker-base",
        {"retriever": "hybrid", "rerank_model": "BAAI/bge-reranker-base"},
    ),
]


# The shipped generation config uses final_k=8 (what the Synthesizer reads).
# The retrieval ladder needs final_k >= the scoring cutoff, or recall@10 would
# be structurally capped at recall@8 -- so the ablation runs at 20 and the
# difference is recorded in the report rather than left implicit.
LADDER_FINAL_K = 20


def _config(corpus_id: str, **overrides) -> RunConfig:
    return RunConfig(
        ingest=IngestConfig(corpus_id=corpus_id),
        retrieval=RetrievalConfig(**overrides),
    )


def cmd_init(_: argparse.Namespace) -> int:
    apply_schema()
    print(json.dumps(server_info(), indent=2))
    return 0


def cmd_ingest_scifact(args: argparse.Namespace) -> int:
    from agentforge.ingest import beir

    stats = beir.load(limit_docs=args.limit)
    print(json.dumps(stats, indent=2))
    if stats["queries_without_positive_qrel"]:
        print(
            f"note: {stats['queries_without_positive_qrel']} queries have no positive "
            "qrel and will be refused by the scorer rather than averaged in as zero",
            file=sys.stderr,
        )
    return 0


def cmd_ingest_arxiv(args: argparse.Namespace) -> int:
    from agentforge.ingest import papers

    report = papers.ingest(
        IngestConfig(corpus_id=papers.CORPUS_ID),
        n_papers=args.n,
        embed_model=args.embed_model,
    )
    print(json.dumps(report.__dict__, indent=2, default=str))
    print(
        f"\ncontamination check: published {report.published_from} .. "
        f"{report.published_to} (must post-date the generator's training cutoff)"
    )
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    config = _config(args.corpus, embed_model=args.embed_model)
    state = lexical.build_index(config.retrieval)
    print(f"bm25: rebuilt={state.rebuilt} seconds={state.seconds:.2f} analyzer={state.analyzer}")
    embedder = Embedder(config.retrieval.embed_model, config.retrieval.embed_dim)
    stats = embedder.index_corpus(args.corpus)
    print(f"dense ({args.embed_model}): {json.dumps(stats)}")
    return 0


def _latest_finished(arm_label: str, tier: str) -> str | None:
    from agentforge.db.conn import pool

    with pool().connection() as conn:
        row = conn.execute(
            "SELECT run_id FROM runs WHERE arm_label = %s AND tier = %s "
            "AND finished_at IS NOT NULL ORDER BY started_at DESC LIMIT 1",
            (arm_label, tier),
        ).fetchone()
    return row["run_id"] if row else None


def _write_ladder(tier: str, runs: list[tuple[str, str]], k: int, metric: str) -> pathlib.Path:
    steps = report.build_ladder(runs, metric=metric, k=k)
    body = (
        f"# Retrieval ablation — {tier}\n\n"
        f"{report.environment_note()}"
        f"- Scoring cutoff k={k}, pipeline final_k={LADDER_FINAL_K} "
        f"(final_k must be >= k or every arm is capped at recall@final_k)\n\n"
        f"{report.ladder_markdown(steps, k=k, metric=metric)}"
    )
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"retrieval-{tier}.md"
    out.write_text(body)
    print(body)
    return out


def cmd_validate_judge(args: argparse.Namespace) -> int:
    """Score candidate judges against the reference labels and pick one.

    The judge is chosen by measurement rather than by price: running three
    models over ~100 pairs costs about a dollar, and kappa says which of them
    is actually an instrument.
    """
    from agentforge.eval.judge import EntailmentJudge, validate_judge
    from agentforge.llm import LLMClient

    results = []
    for model in args.models:
        judge = EntailmentJudge(LLMClient(model=model, max_tokens=256))
        result = validate_judge(judge, min_kappa=args.min_kappa)
        results.append(result)
        print(result.describe())
        if result.note:
            print(f"    {result.note}")

    usable = [r for r in results if r.usable]
    body = [
        "# Judge validation",
        "",
        "An unvalidated LLM judge makes every groundedness number above it decorative.",
        "Kappa rather than raw agreement, because the label distribution is skewed: a",
        "judge answering \"entailed\" every time posts ~90% agreement with no skill.",
        "",
        "| Judge | prompt | kappa | agreement | n | labels | usable |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        body.append(
            f"| `{r.judge_model}` | {r.prompt_version} | **{r.kappa:.3f}** | "
            f"{r.agreement:.3f} | {r.n_pairs} | {r.label_source} | "
            f"{'yes' if r.usable else 'NO'} |"
        )
    body += ["", f"- Floor: kappa >= {args.min_kappa:.2f}"]
    if usable:
        cheapest = min(usable, key=lambda r: _model_cost_rank(r.judge_model))
        body.append(
            f"- Shipped judge: **{cheapest.judge_model}** -- the cheapest model clearing "
            "the floor, chosen by measurement rather than by price."
        )
    else:
        body.append(
            "- **No judge cleared the floor.** The prompt needs fixing and re-measuring; "
            "numbers computed with a sub-floor judge are not reportable."
        )
    if any(not r.human_validated for r in results):
        body.append(
            "- Reference labels are model-generated and PENDING HUMAN REVIEW, so these "
            "figures measure model self-consistency, not human agreement."
        )

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "judge-validation.md"
    out.write_text("\n".join(body) + "\n")
    print(f"\nwritten to {out}")
    return 0


def _model_cost_rank(model: str) -> float:
    from agentforge.config import MODEL_PRICES

    per_in, per_out = MODEL_PRICES[model]
    return per_in + per_out


def cmd_review_labels(args: argparse.Namespace) -> int:
    """Interactive review of the reference labels.

    This is the one task in the project that cannot be automated away: if the
    same model both writes the labels and answers as the judge, kappa measures
    self-consistency and calling it human agreement would be a lie.
    """
    from agentforge.eval.judge import HUMAN_REVIEWED, mark_reviewed, reference_pairs

    pending = [p for p in reference_pairs() if p["label_source"] != HUMAN_REVIEWED]
    if not pending:
        print("all reference pairs are human-reviewed.")
        return 0

    print(f"{len(pending)} pair(s) pending review.")
    print("For each: [e]ntailed  [c]ontradicted  [u]nsupported  [k]eep as-is  [q]uit\n")
    choices = {"e": "entailed", "c": "contradicted", "u": "unsupported"}
    for index, pair in enumerate(pending, start=1):
        print(f"--- {index}/{len(pending)}  ({pair['pair_id']})")
        print(f"CLAIM: {pair['claim_text']}")
        print(f"QUOTE: {pair['quote']}")
        print(f"model label: {pair['label']}")
        answer = input("your verdict> ").strip().lower()
        if answer == "q":
            break
        if answer == "k":
            mark_reviewed(pair["pair_id"], pair["label"])
        elif answer in choices:
            mark_reviewed(pair["pair_id"], choices[answer])
        else:
            print("  unrecognised; left pending")
        print()
    return 0


def cmd_measure_parser(args: argparse.Namespace) -> int:
    """Score the PDF parser against arXiv's own LaTeXML rendering.

    Two metrics, because they fail differently: section recall/precision says
    how well heading detection agrees with the author's sectioning, and text
    coverage says whether any text is being lost. A missed heading merges a
    section into its predecessor -- coarser citations, nothing unretrievable --
    while uncovered text is never chunked at all.

    A paper whose rendering cannot be fetched is a COVERAGE GAP in the
    measurement, counted and reported. Text coverage needs no network and is
    reported for every paper regardless.
    """
    from agentforge.errors import IngestError
    from agentforge.ingest.fidelity import (
        fetch_latexml_html,
        headings_from_latexml,
        run_in_headings_from_latexml,
        score_sections,
        text_coverage,
    )
    from agentforge.ingest.pdf import parse_pdf

    pdfs = sorted(pathlib.Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs in {args.pdf_dir}")

    rows: list[dict[str, object]] = []
    for path in pdfs:
        parsed = parse_pdf(path)
        found = [section.section_path for section in parsed.sections]
        row: dict[str, object] = {
            "paper": path.stem,
            "pages": parsed.n_pages,
            "found": len(found),
            "coverage": text_coverage(parsed.sections, len(parsed.text)),
            "recall": None,
            "precision": None,
            "truth": 0,
            "run_in": 0,
            "gap": "",
        }
        try:
            html = fetch_latexml_html(path.stem)
        except IngestError as exc:
            html, row["gap"] = None, f"fetch failed: {str(exc)[:60]}"
        if html is not None:
            scores = score_sections(found, headings_from_latexml(html))
            row.update(
                recall=scores.recall,
                precision=scores.precision,
                truth=scores.n_truth,
                run_in=len(run_in_headings_from_latexml(html)),
            )
            if scores.recall is None:
                row["gap"] = "no section tags in rendering"
        elif not row["gap"]:
            row["gap"] = "no LaTeXML rendering"
        rows.append(row)
        print(
            f"{row['paper']:<14} coverage={row['coverage']:.3f} "
            + (
                f"recall={row['recall']:.3f} precision={row['precision']:.3f} "
                f"(truth {row['truth']}, found {row['found']})"
                if row["recall"] is not None
                else f"[{row['gap']}] found {row['found']}"
            )
        )

    scored = [r for r in rows if r["recall"] is not None]
    body = [
        "# PDF parser fidelity",
        "",
        "Ground truth is arXiv's own LaTeXML rendering (`arxiv.org/html/<id>`), so",
        "the reference sectioning comes from the authors' LaTeX rather than from how",
        "the PDF happens to look. No labelling, no cost.",
        "",
        "**Read the two metrics separately.** A missed heading merges its body into the",
        "previous section: citations get coarser, but every character is still chunked",
        "and retrievable. Uncovered text is the failure that would actually invalidate a",
        "retrieval number. Coverage below 1.0 is expected -- title, authors and",
        "affiliations precede the first heading and belong to no section.",
        "",
        "Run-in `\\paragraph{}` labels are counted but NOT scored: they render inline",
        "with their prose, so a line-based parser cannot see them. On one paper they are",
        "25 of 38 LaTeXML headings, and including them would have reported the metric's",
        "definition rather than the parser's quality.",
        "",
        "| Paper | pages | text coverage | section recall | section precision | truth | found | run-in (unscored) | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        recall = f"{r['recall']:.3f}" if r["recall"] is not None else "—"
        precision = f"{r['precision']:.3f}" if r["precision"] is not None else "—"
        body.append(
            f"| {r['paper']} | {r['pages']} | {r['coverage']:.3f} | {recall} | "
            f"{precision} | {r['truth'] or '—'} | {r['found']} | {r['run_in'] or '—'} | "
            f"{r['gap'] or ''} |"
        )
    body.append("")
    mean_coverage = sum(float(r["coverage"]) for r in rows) / len(rows)
    body.append(f"- Papers measured: {len(rows)}; with ground truth: {len(scored)}")
    body.append(f"- Mean text coverage: **{mean_coverage:.3f}**")
    if scored:
        body.append(
            f"- Mean section recall: **{sum(float(r['recall']) for r in scored)/len(scored):.3f}**"
        )
        body.append(
            "- Mean section precision: "
            f"**{sum(float(r['precision']) for r in scored)/len(scored):.3f}**"
        )
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "parser-fidelity.md"
    out.write_text("\n".join(body) + "\n")
    print()
    print("\n".join(body[-4:]))
    print(f"written to {out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Regenerate the ladder table from stored runs, without re-running anything."""
    runs: list[tuple[str, str]] = []
    for label, _ in LADDER:
        run_id = _latest_finished(label, args.tier)
        if run_id is None:
            raise SystemExit(
                f"no finished run stored for arm {label!r} on tier {args.tier!r}; "
                "run eval-retrieval first"
            )
        runs.append((label, run_id))
    print(f"written to {_write_ladder(args.tier, runs, args.k, args.metric)}")
    return 0


def cmd_eval_retrieval(args: argparse.Namespace) -> int:
    runs: list[tuple[str, str]] = []
    for label, overrides in LADDER:
        config = _config(
            args.corpus,
            embed_model=args.embed_model,
            final_k=LADDER_FINAL_K,
            **overrides,
        )
        lexical.build_index(config.retrieval)
        handle = harness.execute_retrieval_run(
            config, tier=args.tier, arm_label=label, cache_bypass=True
        )
        scores = harness.score_run(handle.run_id, k=args.k)
        print(f"{label:<34} {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()})}")
        runs.append((label, handle.run_id))

    print(f"written to {_write_ladder(args.tier, runs, args.k, args.metric)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentforge")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="apply schema, print engine versions").set_defaults(
        func=cmd_init
    )

    p = sub.add_parser("ingest-scifact", help="load the BEIR SciFact tier")
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_ingest_scifact)

    p = sub.add_parser("ingest-arxiv", help="fetch, parse, chunk and store arXiv papers")
    p.add_argument("-n", type=int, default=300, help="number of papers")
    p.add_argument("--embed-model", dest="embed_model", default="BAAI/bge-base-en-v1.5")
    p.set_defaults(func=cmd_ingest_arxiv)

    p = sub.add_parser("index", help="build the BM25 and dense indexes")
    p.add_argument("--corpus", default="scifact")
    p.add_argument("--embed-model", dest="embed_model", default="BAAI/bge-base-en-v1.5")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("eval-retrieval", help="run the ladder and write the table")
    p.add_argument("--corpus", default="scifact")
    p.add_argument("--tier", default="scifact")
    p.add_argument("--embed-model", dest="embed_model", default="BAAI/bge-base-en-v1.5")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--metric", default="ndcg", choices=("recall", "ndcg", "mrr"))
    p.set_defaults(func=cmd_eval_retrieval)

    p = sub.add_parser("validate-judge", help="score candidate judges, report kappa")
    p.add_argument(
        "--models", nargs="+",
        default=["claude-haiku-4-5", "claude-sonnet-5", "claude-opus-5"],
    )
    p.add_argument("--min-kappa", dest="min_kappa", type=float, default=0.70)
    p.set_defaults(func=cmd_validate_judge)

    p = sub.add_parser("review-labels", help="review reference labels by hand")
    p.set_defaults(func=cmd_review_labels)

    p = sub.add_parser("measure-parser", help="score the PDF parser against arXiv HTML")
    p.add_argument("--pdf-dir", dest="pdf_dir", default="data/pdfs")
    p.set_defaults(func=cmd_measure_parser)

    p = sub.add_parser("report", help="regenerate the table from stored runs")
    p.add_argument("--tier", default="scifact")
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--metric", default="ndcg", choices=("recall", "ndcg", "mrr"))
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    finally:
        close_pool()


if __name__ == "__main__":
    raise SystemExit(main())
