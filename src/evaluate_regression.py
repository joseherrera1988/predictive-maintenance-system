"""
Regression metrics for RUL prediction on CMAPSS.

Includes the standard NASA asymmetric scoring function used in the
PHM08 competition, which penalises late predictions more heavily.
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA PHM08 asymmetric scoring function.
    Late predictions (d > 0) are penalised more than early ones (d < 0).

        S = sum(exp(-d/13) - 1)  for d < 0
          + sum(exp( d/10) - 1)  for d >= 0

    where d = y_pred - y_true
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    score = nasa_score(y_true, y_pred)
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4),
        "nasa_score": round(score, 2),
    }
