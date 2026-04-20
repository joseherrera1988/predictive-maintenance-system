import numpy as np
import pandas as pd
import pytest

from src.nested_cv import nested_cv_random_forest, nested_cv_xgboost

N_ENGINES = 10
N_CYCLES = 40
SENSOR_COLS = ["s2", "s3", "s4"]
EXPECTED_KEYS = {"precision", "recall", "roc_auc", "y_true_bin", "y_pred_bin", "best_params_per_fold"}
N_FOLDS = 2
N_INNER_FOLDS = 2
N_TRIALS = 3


@pytest.fixture(scope="module")
def tiny_train_file(tmp_path_factory):
    """Write a minimal CMAPSS-format train file for smoke testing."""
    tmp = tmp_path_factory.mktemp("data")
    path = tmp / "train_tiny.txt"
    cols = (
        ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
        + [f"s{i}" for i in range(1, 22)]
    )
    rows = []
    for uid in range(1, N_ENGINES + 1):
        for cycle in range(1, N_CYCLES + 1):
            row = [uid, cycle, 0.0, 0.0, 100.0] + [float(uid + cycle * 0.1)] * 21
            rows.append(row)
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, sep=" ", index=False, header=False)
    return str(path)


@pytest.fixture(scope="module")
def rf_result(tiny_train_file):
    return nested_cv_random_forest(
        tiny_train_file,
        n_folds=N_FOLDS,
        n_trials=N_TRIALS,
        n_inner_folds=N_INNER_FOLDS,
        random_state=0,
    )


@pytest.fixture(scope="module")
def xgb_result(tiny_train_file):
    return nested_cv_xgboost(
        tiny_train_file,
        n_folds=N_FOLDS,
        n_trials=N_TRIALS,
        n_inner_folds=N_INNER_FOLDS,
        random_state=0,
    )


def test_nested_cv_rf_returns_expected_keys(rf_result):
    assert EXPECTED_KEYS.issubset(rf_result.keys())


def test_nested_cv_rf_fold_count(rf_result):
    assert len(rf_result["precision"]) == N_FOLDS
    assert len(rf_result["recall"]) == N_FOLDS
    assert len(rf_result["roc_auc"]) == N_FOLDS


def test_nested_cv_xgb_returns_expected_keys(xgb_result):
    assert EXPECTED_KEYS.issubset(xgb_result.keys())


def test_nested_cv_xgb_fold_count(xgb_result):
    assert len(xgb_result["precision"]) == N_FOLDS
    assert len(xgb_result["recall"]) == N_FOLDS
    assert len(xgb_result["roc_auc"]) == N_FOLDS


def test_best_params_per_fold_length(rf_result, xgb_result):
    assert len(rf_result["best_params_per_fold"]) == N_FOLDS
    assert all(isinstance(p, dict) for p in rf_result["best_params_per_fold"])
    assert len(xgb_result["best_params_per_fold"]) == N_FOLDS
    assert all(isinstance(p, dict) for p in xgb_result["best_params_per_fold"])
