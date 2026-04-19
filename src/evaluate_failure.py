"""
Classification metrics for failure prediction derived from RUL estimates.
Binarizes continuous RUL values using a threshold: failure = (RUL <= threshold).
"""

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

FAILURE_THRESHOLD = 30  # cycles; engines with RUL <= this are "near failure"


def evaluate_failure(
    y_true_rul: np.ndarray,
    y_pred_rul: np.ndarray,
    threshold: int = FAILURE_THRESHOLD,
) -> dict:
    y_true_bin = (y_true_rul <= threshold).astype(int)
    y_pred_bin = (y_pred_rul <= threshold).astype(int)

    # Use predicted RUL distance from threshold as a continuous score for ROC-AUC
    # (lower predicted RUL → higher probability of failure)
    y_score = -y_pred_rul

    try:
        auc = round(roc_auc_score(y_true_bin, y_score), 4)
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true_bin, y_pred_bin)

    return {
        "threshold": threshold,
        "recall": round(recall_score(y_true_bin, y_pred_bin, zero_division=0), 4),
        "precision": round(precision_score(y_true_bin, y_pred_bin, zero_division=0), 4),
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


def print_failure_metrics(metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    print(f"\n[CLS] Failure Classification (threshold={metrics['threshold']} cycles)")
    print(f"  Recall     : {metrics['recall']}  (primary metric)")
    print(f"  Precision  : {metrics['precision']}")
    print(f"  ROC-AUC    : {metrics['roc_auc']}")
    print( "  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")
