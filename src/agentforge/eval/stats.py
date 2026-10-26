"""Statistics for small, paired evaluation sets.

Two facts drive everything here.

First, N is small on the tiers that matter. At N=40 a difference of five points
is inside the noise unless the comparison is paired, so every arm runs the
identical frozen question set and differences are computed per question. Pairing
is not a nicety; it is what makes the tier usable at all.

Second, a point estimate without an interval invites exactly the overclaim this
project is built to avoid. "Reranking improved nDCG by 4.2 points" and
"reranking improved nDCG by 4.2 points, 95% CI [-0.8, 9.1]" are different
claims, and only one of them is honest at N=40.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedDelta:
    mean_delta: float
    ci_low: float
    ci_high: float
    n: int
    n_better: int
    n_worse: int
    n_tied: int
    significant: bool

    def describe(self, metric: str = "metric") -> str:
        direction = "improved" if self.mean_delta >= 0 else "regressed"
        marker = "" if self.significant else " (not significant)"
        return (
            f"{metric} {direction} by {self.mean_delta:+.4f} "
            f"[95% CI {self.ci_low:+.4f}, {self.ci_high:+.4f}], n={self.n}{marker}"
        )


def paired_bootstrap(
    baseline: dict[str, float],
    treatment: dict[str, float],
    iterations: int = 10_000,
    seed: int = 20261026,
    confidence: float = 0.95,
) -> PairedDelta:
    """Bootstrap the mean paired difference, resampling questions with replacement.

    Both arms must cover exactly the same question ids -- a treatment evaluated
    on a different subset is not a paired comparison, and averaging over
    different denominators is how a regression hides.
    """
    if set(baseline) != set(treatment):
        only_base = sorted(set(baseline) - set(treatment))[:3]
        only_treat = sorted(set(treatment) - set(baseline))[:3]
        raise ValueError(
            "paired comparison requires identical question sets; "
            f"baseline-only={only_base} treatment-only={only_treat}"
        )
    if not baseline:
        raise ValueError("no questions to compare")

    keys = sorted(baseline)
    deltas = [treatment[k] - baseline[k] for k in keys]
    n = len(deltas)
    mean_delta = sum(deltas) / n

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()

    tail = (1.0 - confidence) / 2.0
    ci_low = means[int(tail * iterations)]
    ci_high = means[min(int((1.0 - tail) * iterations), iterations - 1)]

    return PairedDelta(
        mean_delta=mean_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        n_better=sum(1 for d in deltas if d > 0),
        n_worse=sum(1 for d in deltas if d < 0),
        n_tied=sum(1 for d in deltas if d == 0),
        # The interval excluding zero is the condition for calling a delta real.
        significant=(ci_low > 0.0 or ci_high < 0.0),
    )


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. Used for p95 latency.

    No interpolation: with ~120 observations, interpolating between the two
    slowest samples invents precision the sample size does not support.
    """
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    index = min(int(round(q / 100.0 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def cohens_kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    """Cohen's kappa and raw agreement between two label sequences.

    This is what licenses the LLM judge as an instrument. Raw agreement alone is
    misleading when the label distribution is skewed -- and it is skewed here,
    because most claims in a working system are entailed, so a judge that
    answered "entailed" every time would post ~85% agreement and kappa near
    zero. Kappa is the number that catches that.
    """
    if len(a) != len(b):
        raise ValueError(f"label sequences differ in length: {len(a)} vs {len(b)}")
    if not a:
        raise ValueError("no labelled pairs")

    n = len(a)
    labels = sorted(set(a) | set(b))
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    expected = sum(
        (a.count(label) / n) * (b.count(label) / n) for label in labels
    )
    if expected == 1.0:
        # Both raters used exactly one label for everything; kappa is undefined
        # rather than perfect, and reporting 1.0 would be a lie.
        raise ValueError(
            "kappa undefined: both raters assigned a single label to every pair"
        )
    return (observed - expected) / (1.0 - expected), observed
