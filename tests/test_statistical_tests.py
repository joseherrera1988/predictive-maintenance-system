import numpy as np
import pytest
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon

from src.statistical_tests import (
    compute_summary,
    friedman_test,
    mcnemar_test,
    pairwise_ttest,
    pairwise_wilcoxon,
)

A = [0.90, 0.85, 0.88, 0.92, 0.87]
B = [0.80, 0.75, 0.82, 0.78, 0.79]
C = [0.95, 0.91, 0.93, 0.96, 0.94]


def test_summary_mean_std():
    summary = compute_summary({"A": A, "B": B})
    assert abs(summary["A"]["mean"] - np.mean(A)) < 1e-9
    assert abs(summary["A"]["std"] - np.std(A, ddof=1)) < 1e-9
    assert summary["A"]["n"] == 5
    assert summary["B"]["n"] == 5


def test_friedman_matches_scipy():
    result = friedman_test(A, B, C)
    expected_stat, expected_p = friedmanchisquare(A, B, C)
    assert abs(result["statistic"] - expected_stat) < 1e-9
    assert abs(result["p_value"] - expected_p) < 1e-9


def test_wilcoxon_p_value_matches_scipy():
    result = pairwise_wilcoxon(A, B, n_comparisons=1)
    _, p_expected = wilcoxon(A, B, alternative="two-sided")
    assert abs(result["p_value"] - p_expected) < 1e-9


def test_wilcoxon_bonferroni_correction():
    result = pairwise_wilcoxon(A, B, n_comparisons=3)
    _, p_raw = wilcoxon(A, B, alternative="two-sided")
    assert abs(result["p_corrected"] - min(p_raw * 3, 1.0)) < 1e-9


def test_wilcoxon_p_corrected_capped_at_one():
    result = pairwise_wilcoxon(A, B, n_comparisons=1000)
    assert result["p_corrected"] <= 1.0


def test_wilcoxon_effect_size_range():
    result = pairwise_wilcoxon(A, B, n_comparisons=1)
    assert -1.0 <= result["effect_size_r"] <= 1.0


def test_ttest_matches_scipy():
    result = pairwise_ttest(A, B, n_comparisons=1)
    _, p_expected = ttest_rel(A, B)
    assert abs(result["p_value"] - p_expected) < 1e-9


def test_ttest_bonferroni_correction():
    result = pairwise_ttest(A, B, n_comparisons=3)
    _, p_raw = ttest_rel(A, B)
    assert abs(result["p_corrected"] - min(p_raw * 3, 1.0)) < 1e-9


def test_mcnemar_contingency_counts():
    # y_true=[1,1,1,0,0], pred_a=[1,1,0,0,1], pred_b=[1,0,1,0,1]
    # correct_a: T,T,F,T,F → indices 0,1,3
    # correct_b: T,F,T,T,F → indices 0,2,3
    # n10 (A right, B wrong): index 1  → 1
    # n01 (A wrong, B right): index 2  → 1
    result = mcnemar_test([1, 1, 1, 0, 0], [1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert result["n10"] == 1
    assert result["n01"] == 1


def test_mcnemar_identical_predictions_p_is_nan_or_one():
    # When predictions are identical, McNemar statistic = 0
    y_true = [1, 0, 1, 0, 1]
    pred = [1, 0, 1, 0, 1]
    result = mcnemar_test(y_true, pred, pred)
    # n10 == n01 == 0 → statistic = 0
    assert result["n10"] == 0
    assert result["n01"] == 0


def test_mcnemar_returns_required_keys():
    result = mcnemar_test([1, 0, 1], [1, 1, 0], [0, 1, 1])
    for key in ("statistic", "p_value", "n01", "n10"):
        assert key in result
