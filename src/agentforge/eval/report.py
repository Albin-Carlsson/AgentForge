"""Generated reports. The tables in the README are build artefacts, not prose.

A hand-assembled results table drifts from the database the moment anything is
re-run, and a stale number is worse than no number. So the ablation table is
produced from SQL, written to results/, and the README points at it.

The table is a LADDER: each row differs from the row above by exactly one knob,
with a paired-bootstrap delta attached. That shape is forced by
`config.assert_comparable` -- comparing "dense only" against "hybrid + rerank"
moves two knobs at once and cannot be attributed to either, so the harness will
not produce that row.
"""

from __future__ import annotations

from dataclasses import dataclass

from tabulate import tabulate

from agentforge.eval import harness


@dataclass(frozen=True)
class LadderStep:
    arm_label: str
    run_id: str
    scores: dict[str, float | int | str]
    knob: str | None
    delta: str | None


def build_ladder(
    runs: list[tuple[str, str]], metric: str, k: int
) -> list[LadderStep]:
    """`runs` is [(arm_label, run_id), ...] in ladder order, baseline first."""
    steps: list[LadderStep] = []
    for index, (label, run_id) in enumerate(runs):
        knob: str | None = None
        delta: str | None = None
        if index > 0:
            comparison = harness.compare_runs(
                runs[index - 1][1], run_id, metric=metric, k=k
            )
            knob = comparison.knob
            paired = comparison.delta
            marker = "" if paired.significant else " ns"
            delta = (
                f"{paired.mean_delta:+.4f} "
                f"[{paired.ci_low:+.4f}, {paired.ci_high:+.4f}]{marker}"
            )
        steps.append(
            LadderStep(
                arm_label=label,
                run_id=run_id,
                scores=harness.score_run(run_id, k=k),
                knob=knob,
                delta=delta,
            )
        )
    return steps


def ladder_markdown(steps: list[LadderStep], k: int, metric: str) -> str:
    headers = [
        "Arm", f"recall@{k}", f"nDCG@{k}", "MRR", "n", "knob changed",
        f"paired Δ {metric}@{k} (95% CI)",
    ]
    rows = [
        [
            step.arm_label,
            f"{step.scores[f'recall@{k}']:.4f}",
            f"{step.scores[f'nDCG@{k}']:.4f}",
            f"{step.scores['MRR']:.4f}",
            step.scores["n_queries"],
            step.knob or "— (baseline)",
            step.delta or "—",
        ]
        for step in steps
    ]
    table = tabulate(rows, headers=headers, tablefmt="github")
    return (
        f"{table}\n\n"
        "Each row differs from the row above by exactly one knob; `assert_comparable`\n"
        "refuses any comparison that moves two. `ns` marks a delta whose 95% paired\n"
        "bootstrap interval includes zero -- i.e. not distinguishable from noise at\n"
        f"n={steps[0].scores['n_queries'] if steps else 0}.\n"
    )


def environment_note() -> str:
    from agentforge.db.conn import server_info
    from agentforge.retrieval import lexical
    from agentforge.retrieval.embed import device

    info = server_info()
    bm25 = lexical.bm25_parameters()
    return (
        f"- Engine: {info.get('postgres', 'unknown')}, "
        f"pg_search {info.get('pg_search', '?')}, pgvector {info.get('vector', '?')}\n"
        f"- BM25 constants: k1={bm25['k1']}, b={bm25['b']} "
        f"({bm25['source']}, tunable: {bm25['tunable']})\n"
        f"- Embedding/rerank device: {device()}\n"
    )
