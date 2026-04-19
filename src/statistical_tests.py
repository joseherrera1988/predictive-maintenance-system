"""
Statistical significance tests for comparing ML model performance.

All pairwise tests apply Bonferroni correction. Pass n_comparisons=3 for
three-model comparisons (RF vs XGBoost, RF vs LSTM, XGBoost vs LSTM).
"""

import numpy as np
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar as _mcnemar_fn


def compute_summary(scores_by_model: dict) -> dict:
    """Returns mean, std (ddof=1), and n per model."""
    return {
        model: {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)),
            "n": len(scores),
        }
        for model, scores in scores_by_model.items()
    }


def friedman_test(*score_lists) -> dict:
    """Friedman chi-square test across 3+ models (non-parametric)."""
    stat, p = friedmanchisquare(*score_lists)
    return {"statistic": float(stat), "p_value": float(p)}


def pairwise_wilcoxon(a: list, b: list, n_comparisons: int = 1) -> dict:
    """
    Wilcoxon signed-rank test with Bonferroni correction.
    Effect size: rank-biserial r (r > 0 means a > b on average).
    """
    n = len(a)
    stat, p = wilcoxon(a, b, alternative="two-sided")
    p_corrected = min(float(p) * n_comparisons, 1.0)
    # Rank-biserial r from Wilcoxon W statistic
    max_w = n * (n + 1) / 2
    r = float(1 - (2 * stat) / max_w)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "p_corrected": p_corrected,
        "effect_size_r": r,
    }


def pairwise_ttest(a: list, b: list, n_comparisons: int = 1) -> dict:
    """
    Paired t-test with Bonferroni correction.
    Assumes normality of pairwise differences — interpret cautiously with small n.
    """
    stat, p = ttest_rel(a, b)
    p_corrected = min(float(p) * n_comparisons, 1.0)
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "p_corrected": p_corrected,
    }


def mcnemar_test(y_true_bin: list, y_pred_bin_a: list, y_pred_bin_b: list) -> dict:
    """
    McNemar test comparing two classifiers on the same samples.
    Pools per-sample predictions across all CV folds.

    Contingency table:
                  B correct  B wrong
      A correct |   n11    |   n10  |
      A wrong   |   n01    |   n00  |

    n10 = A right & B wrong; n01 = A wrong & B right (the discordant pairs).
    """
    y_true = np.array(y_true_bin)
    pred_a = np.array(y_pred_bin_a)
    pred_b = np.array(y_pred_bin_b)

    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    n11 = int(np.sum(correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n00 = int(np.sum(~correct_a & ~correct_b))

    table = [[n11, n10], [n01, n00]]
    result = _mcnemar_fn(table, exact=False, correction=True)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n01": n01,
        "n10": n10,
    }
