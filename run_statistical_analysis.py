#!/usr/bin/env python
"""
Statistical significance analysis for RF, XGBoost, and LSTM model comparison.

Usage:
    python run_statistical_analysis.py --dataset FD001
    python run_statistical_analysis.py --dataset all --output experiments/statistical_report.txt
"""

import argparse
import sys
from pathlib import Path

from src.cross_validate import cv_lstm, cv_random_forest, cv_xgboost
from src.statistical_tests import (
    compute_summary,
    friedman_test,
    mcnemar_test,
    pairwise_ttest,
    pairwise_wilcoxon,
)

DATASETS = {
    "FD001": "data/raw/train_FD001.txt",
    "FD002": "data/raw/train_FD002.txt",
    "FD003": "data/raw/train_FD003.txt",
    "FD004": "data/raw/train_FD004.txt",
}
PAIRS_3 = [("RF", "XGBoost"), ("RF", "LSTM"), ("XGBoost", "LSTM")]
PAIRS_2 = [("RF", "XGBoost")]
ALPHA = 0.05


def _sig(p: float) -> str:
    return "*" if p < ALPHA else " "


def _run_dataset(dataset: str, n_folds: int, cv_epochs: int) -> tuple:
    train_path = DATASETS[dataset]
    lines = []

    sep = "=" * 64
    lines.append("")
    lines.append(sep)
    lines.append(f"Dataset: {dataset}  |  {n_folds}-fold Group CV  |  alpha={ALPHA}")
    lines.append(sep)

    print(f"\n[{dataset}] Random Forest ({n_folds} folds)...")
    rf = cv_random_forest(train_path, n_folds=n_folds)

    print(f"\n[{dataset}] XGBoost ({n_folds} folds)...")
    xgb = cv_xgboost(train_path, n_folds=n_folds)

    print(f"\n[{dataset}] LSTM ({n_folds} folds, {cv_epochs} epochs each)...")
    lstm = cv_lstm(train_path, n_folds=n_folds, cv_epochs=cv_epochs)

    results = {"RF": rf, "XGBoost": xgb}
    active_models = ["RF", "XGBoost"]
    if lstm is not None:
        results["LSTM"] = lstm
        active_models.append("LSTM")
    else:
        lines.append("  [LSTM] Skipped -- TensorFlow not available on this platform.")

    pairs = PAIRS_3 if "LSTM" in results else PAIRS_2
    n_pairs = len(pairs)

    for metric in ("precision", "roc_auc"):
        scores = {m: results[m][metric] for m in active_models}
        summary = compute_summary(scores)

        lines.append("")
        lines.append(f"-- {metric.upper()} --")
        lines.append(f"  {'Model':<10}  {'Mean':>8}  {'Std':>8}  Per-fold scores")
        for m in active_models:
            s = summary[m]
            fold_str = "  ".join(f"{v:.4f}" for v in scores[m])
            lines.append(f"  {m:<10}  {s['mean']:>8.4f}  {s['std']:>8.4f}  [{fold_str}]")

        if len(active_models) >= 3:
            fr = friedman_test(scores["RF"], scores["XGBoost"], scores["LSTM"])
            lines.append("")
            lines.append(
                f"  Friedman (3 models, {n_folds} folds):  "
                f"chi2={fr['statistic']:.4f}  p={fr['p_value']:.4f}{_sig(fr['p_value'])}"
            )
        else:
            lines.append("")
            lines.append("  Friedman: skipped (requires 3+ models)")

        lines.append("")
        lines.append(f"  Pairwise Wilcoxon (Bonferroni k={n_pairs}):")
        for m1, m2 in pairs:
            w = pairwise_wilcoxon(scores[m1], scores[m2], n_pairs)
            lines.append(
                f"    {m1} vs {m2:<10}  W={w['statistic']:>5.1f}  "
                f"p={w['p_value']:.4f}  p_corr={w['p_corrected']:.4f}{_sig(w['p_corrected'])}  "
                f"r={w['effect_size_r']:+.3f}"
            )

        lines.append("")
        lines.append(f"  Paired t-test (Bonferroni k={n_pairs}) [assumes normality]:")
        for m1, m2 in pairs:
            t = pairwise_ttest(scores[m1], scores[m2], n_pairs)
            lines.append(
                f"    {m1} vs {m2:<10}  t={t['statistic']:>+7.4f}  "
                f"p={t['p_value']:.4f}  p_corr={t['p_corrected']:.4f}{_sig(t['p_corrected'])}"
            )

    lines.append("")
    lines.append(f"-- McNemar (pooled per-sample predictions across {n_folds} folds) --")
    for m1, m2 in pairs:
        mc = mcnemar_test(
            results[m1]["y_true_bin"],
            results[m1]["y_pred_bin"],
            results[m2]["y_pred_bin"],
        )
        lines.append(
            f"  {m1} vs {m2:<10}  chi2={mc['statistic']:>7.4f}  p={mc['p_value']:.4f}{_sig(mc['p_value'])}  "
            f"(n10={mc['n10']} {m1}-right-{m2}-wrong  n01={mc['n01']} {m1}-wrong-{m2}-right)"
        )

    lines.append("")
    lines.append(
        f"  Note: * = p < {ALPHA}.  With {n_folds} folds, power is limited -- "
        "non-significant results do not prove equivalence."
    )
    return lines, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FD001",
                        help="FD001 | FD002 | FD003 | FD004 | all  (comma-separated OK)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cv-epochs", type=int, default=20)
    parser.add_argument("--output", default="experiments/statistical_report.txt")
    args = parser.parse_args()

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [d.strip() for d in args.dataset.split(",")]
    unknown = [d for d in datasets if d not in DATASETS]
    if unknown:
        print(f"ERROR: unknown dataset(s): {unknown}.")
        sys.exit(1)

    all_lines = [
        "Statistical Significance Analysis -- Predictive Maintenance System",
        f"Datasets: {', '.join(datasets)}  |  CV folds: {args.folds}  |  LSTM cv-epochs: {args.cv_epochs}",
    ]

    for ds in datasets:
        lines, _ = _run_dataset(ds, args.folds, args.cv_epochs)
        all_lines.extend(lines)

    report = "\n".join(all_lines)
    print("\n\n" + report)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved -> {args.output}")


if __name__ == "__main__":
    main()
