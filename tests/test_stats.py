import pytest

from agentforge.eval.stats import cohens_kappa, paired_bootstrap, percentile


def test_paired_bootstrap_detects_a_consistent_improvement():
    baseline = {f"q{i}": 0.50 for i in range(40)}
    treatment = {f"q{i}": 0.60 for i in range(40)}
    result = paired_bootstrap(baseline, treatment, iterations=2000)
    assert result.mean_delta == pytest.approx(0.10)
    assert result.significant
    assert result.n_better == 40


def test_paired_bootstrap_calls_noise_insignificant():
    baseline = {f"q{i}": 0.5 for i in range(40)}
    treatment = {f"q{i}": 0.5 + (0.3 if i % 2 else -0.3) for i in range(40)}
    result = paired_bootstrap(baseline, treatment, iterations=2000)
    assert result.mean_delta == pytest.approx(0.0, abs=1e-9)
    assert not result.significant
    assert result.ci_low < 0 < result.ci_high


def test_mismatched_question_sets_raise():
    with pytest.raises(ValueError, match="identical question sets"):
        paired_bootstrap({"a": 1.0}, {"b": 1.0})


def test_percentile_is_nearest_rank():
    values = [float(i) for i in range(1, 101)]
    assert percentile(values, 95) == 95.0
    assert percentile(values, 50) == 51.0
    assert percentile([5.0], 95) == 5.0


def test_kappa_is_zero_for_chance_agreement():
    # Rater b is a shifted copy: they agree only as often as chance predicts.
    a = ["entailed", "unsupported"] * 50
    b = ["unsupported", "entailed"] * 50
    kappa, agreement = cohens_kappa(a, b)
    assert agreement == 0.0
    assert kappa == pytest.approx(-1.0)


def test_kappa_rewards_agreement_beyond_chance():
    a = ["entailed"] * 80 + ["unsupported"] * 20
    b = ["entailed"] * 78 + ["unsupported"] * 2 + ["unsupported"] * 20
    kappa, agreement = cohens_kappa(a, b)
    assert agreement == pytest.approx(0.98)
    assert kappa > 0.9


def test_kappa_undefined_when_a_rater_uses_one_label_for_everything():
    """A judge answering 'entailed' always would post high agreement; refuse it."""
    a = ["entailed"] * 100
    b = ["entailed"] * 100
    with pytest.raises(ValueError, match="undefined"):
        cohens_kappa(a, b)
