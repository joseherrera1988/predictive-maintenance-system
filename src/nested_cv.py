"""
Nested cross-validation with Optuna hyperparameter tuning for RF and XGBoost.

Outer loop: GroupKFold on unit_id for unbiased metric estimation.
Inner loop: Optuna TPE search on the outer fold's training data, optimizing recall.
Final model per outer fold: retrained on full outer training split with best params.

Return dict is drop-in compatible with cv_random_forest / cv_xgboost.
"""

import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import optuna.logging as optuna_logging

from src.cmapss_loader import get_feature_columns, load_train
from src.config import CONFIG
from src.evaluate_failure import evaluate_failure
from src.feature_engineering import add_time_series_features

optuna_logging.set_verbosity(optuna_logging.WARNING)

FAILURE_THRESHOLD = 30


def _binarize(y_true_rul: np.ndarray, y_pred_rul: np.ndarray):
    return (
        (y_true_rul <= FAILURE_THRESHOLD).astype(int),
        (y_pred_rul <= FAILURE_THRESHOLD).astype(int),
    )


def _load_and_enrich(train_path: str):
    train_df = load_train(train_path)
    _ts = CONFIG.get("time_series_features", {})
    train_df = add_time_series_features(
        train_df,
        windows=tuple(_ts.get("windows", [5, 10, 20])),
        lags=tuple(_ts.get("lags", [1, 5, 10])),
        ewm_span=_ts.get("ewm_span", 10),
    )
    return train_df, get_feature_columns(train_df)


def nested_cv_random_forest(
    train_path: str,
    n_folds: int = 5,
    n_trials: int = 50,
    n_inner_folds: int = 3,
    random_state: int = 42,
) -> dict:
    train_df, feat_cols = _load_and_enrich(train_path)
    groups = train_df["unit_id"].values

    outer_gkf = GroupKFold(n_splits=n_folds)
    precision_scores, recall_scores, roc_auc_scores = [], [], []
    all_y_true_bin, all_y_pred_bin = [], []
    best_params_per_fold = []

    for outer_idx, (outer_train_idx, outer_val_idx) in enumerate(
        outer_gkf.split(train_df, groups=groups)
    ):
        outer_train = train_df.iloc[outer_train_idx]
        outer_val = train_df.iloc[outer_val_idx]
        inner_groups = outer_train["unit_id"].values

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", 0.5]
                ),
                "random_state": random_state,
                "n_jobs": -1,
            }
            inner_gkf = GroupKFold(n_splits=n_inner_folds)
            inner_recalls = []
            for inner_train_idx, inner_val_idx in inner_gkf.split(
                outer_train, groups=inner_groups
            ):
                inner_train = outer_train.iloc[inner_train_idx]
                inner_val = outer_train.iloc[inner_val_idx]
                scaler = StandardScaler()
                X_inner_train = scaler.fit_transform(inner_train[feat_cols])
                X_inner_val = scaler.transform(inner_val[feat_cols])
                y_inner_train = inner_train["RUL"].values
                y_inner_val = inner_val["RUL"].values
                model = RandomForestRegressor(**params)
                model.fit(X_inner_train, y_inner_train)
                y_pred = model.predict(X_inner_val)
                m = evaluate_failure(y_inner_val, y_pred)
                inner_recalls.append(m["recall"])
            return float(np.mean(inner_recalls))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state + outer_idx),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params_per_fold.append(best_params)

        scaler = StandardScaler()
        X_outer_train = scaler.fit_transform(outer_train[feat_cols])
        y_outer_train = outer_train["RUL"].values
        X_outer_val = scaler.transform(outer_val[feat_cols])
        y_outer_val = outer_val["RUL"].values

        final_model = RandomForestRegressor(
            **best_params, random_state=random_state, n_jobs=-1
        )
        final_model.fit(X_outer_train, y_outer_train)
        y_pred = final_model.predict(X_outer_val)

        m = evaluate_failure(y_outer_val, y_pred)
        precision_scores.append(m["precision"])
        recall_scores.append(m["recall"])
        roc_auc_scores.append(m["roc_auc"])

        y_true_bin, y_pred_bin = _binarize(y_outer_val, y_pred)
        all_y_true_bin.extend(y_true_bin.tolist())
        all_y_pred_bin.extend(y_pred_bin.tolist())

        print(
            f"  RF     outer fold {outer_idx+1}/{n_folds} | "
            f"best_recall_inner={study.best_value:.4f} | "
            f"outer: precision={m['precision']:.4f}  recall={m['recall']:.4f}  roc_auc={m['roc_auc']:.4f}"
        )

    return {
        "precision": precision_scores,
        "recall": recall_scores,
        "roc_auc": roc_auc_scores,
        "y_true_bin": all_y_true_bin,
        "y_pred_bin": all_y_pred_bin,
        "best_params_per_fold": best_params_per_fold,
    }


def nested_cv_xgboost(
    train_path: str,
    n_folds: int = 5,
    n_trials: int = 50,
    n_inner_folds: int = 3,
    random_state: int = 42,
) -> dict:
    train_df, feat_cols = _load_and_enrich(train_path)
    groups = train_df["unit_id"].values

    outer_gkf = GroupKFold(n_splits=n_folds)
    precision_scores, recall_scores, roc_auc_scores = [], [], []
    all_y_true_bin, all_y_pred_bin = [], []
    best_params_per_fold = []

    for outer_idx, (outer_train_idx, outer_val_idx) in enumerate(
        outer_gkf.split(train_df, groups=groups)
    ):
        outer_train = train_df.iloc[outer_train_idx]
        outer_val = train_df.iloc[outer_val_idx]
        inner_groups = outer_train["unit_id"].values

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": 0,
            }
            inner_gkf = GroupKFold(n_splits=n_inner_folds)
            inner_recalls = []
            for inner_train_idx, inner_val_idx in inner_gkf.split(
                outer_train, groups=inner_groups
            ):
                inner_train = outer_train.iloc[inner_train_idx]
                inner_val = outer_train.iloc[inner_val_idx]
                scaler = StandardScaler()
                X_inner_train = scaler.fit_transform(inner_train[feat_cols])
                X_inner_val = scaler.transform(inner_val[feat_cols])
                y_inner_train = inner_train["RUL"].values
                y_inner_val = inner_val["RUL"].values
                model = XGBRegressor(**params)
                model.fit(X_inner_train, y_inner_train)
                y_pred = np.clip(model.predict(X_inner_val), 0, None)
                m = evaluate_failure(y_inner_val, y_pred)
                inner_recalls.append(m["recall"])
            return float(np.mean(inner_recalls))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state + outer_idx),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_params_per_fold.append(best_params)

        scaler = StandardScaler()
        X_outer_train = scaler.fit_transform(outer_train[feat_cols])
        y_outer_train = outer_train["RUL"].values
        X_outer_val = scaler.transform(outer_val[feat_cols])
        y_outer_val = outer_val["RUL"].values

        final_model = XGBRegressor(**best_params, random_state=random_state, n_jobs=-1, verbosity=0)
        final_model.fit(X_outer_train, y_outer_train)
        y_pred = np.clip(final_model.predict(X_outer_val), 0, None)

        m = evaluate_failure(y_outer_val, y_pred)
        precision_scores.append(m["precision"])
        recall_scores.append(m["recall"])
        roc_auc_scores.append(m["roc_auc"])

        y_true_bin, y_pred_bin = _binarize(y_outer_val, y_pred)
        all_y_true_bin.extend(y_true_bin.tolist())
        all_y_pred_bin.extend(y_pred_bin.tolist())

        print(
            f"  XGBoost outer fold {outer_idx+1}/{n_folds} | "
            f"best_recall_inner={study.best_value:.4f} | "
            f"outer: precision={m['precision']:.4f}  recall={m['recall']:.4f}  roc_auc={m['roc_auc']:.4f}"
        )

    return {
        "precision": precision_scores,
        "recall": recall_scores,
        "roc_auc": roc_auc_scores,
        "y_true_bin": all_y_true_bin,
        "y_pred_bin": all_y_pred_bin,
        "best_params_per_fold": best_params_per_fold,
    }
