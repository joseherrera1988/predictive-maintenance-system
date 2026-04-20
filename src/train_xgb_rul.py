"""
XGBoost regressor for CMAPSS RUL prediction.
Row-level prediction using the last observed cycle per engine as features.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.cmapss_loader import load_train, load_test, get_feature_columns
from src.evaluate_regression import evaluate_regression
from src.evaluate_failure import evaluate_failure, print_failure_metrics
from src.feature_engineering import add_time_series_features
from src.config import CONFIG
from src.model_utils import save_model
from src.tracker import log_experiment


def train(
    train_path: str = "data/raw/train_FD001.txt",
    test_path: str = "data/raw/test_FD001.txt",
    rul_path: str = "data/raw/RUL_FD001.txt",
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> XGBRegressor:

    train_df = load_train(train_path)
    test_df = load_test(test_path, rul_path)

    _ts = CONFIG.get("time_series_features", {})
    train_df = add_time_series_features(
        train_df,
        windows=tuple(_ts.get("windows", [5, 10, 20])),
        lags=tuple(_ts.get("lags", [1, 5, 10])),
        ewm_span=_ts.get("ewm_span", 10),
    )
    feat_cols = get_feature_columns(train_df)

    X_train = train_df[feat_cols].values
    y_train = train_df["RUL"].values

    for col in feat_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0
    X_test = test_df[feat_cols].values
    y_test = test_df["RUL"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "random_state": random_state,
        "n_jobs": -1,
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    y_pred = np.clip(model.predict(X_test), 0, None)
    metrics = evaluate_regression(y_test, y_pred)

    print("\n[ML] XGBoost — FD001 Results")
    print(f"  RMSE       : {metrics['rmse']}")
    print(f"  MAE        : {metrics['mae']}")
    print(f"  R²         : {metrics['r2']}")
    print(f"  NASA Score : {metrics['nasa_score']}")

    cls_metrics = evaluate_failure(y_test, y_pred)
    print_failure_metrics(cls_metrics)

    save_model(model, "xgb_rul_fd001")
    log_experiment("xgboost_regressor", params, metrics, notes="CMAPSS FD001")

    return model
