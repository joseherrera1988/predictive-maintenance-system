"""
Group K-Fold cross-validation wrappers for RF, XGBoost, and LSTM.

Splits engines (unit_id) into k folds; trains on k-1 groups, validates on 1.
Preserves temporal structure within each engine's cycle history.

Validation uses ALL cycles from held-out engines (not just the last cycle).
This ensures both failure and non-failure classes are present in each fold,
making precision and ROC-AUC well-defined.

Per fold:
  - StandardScaler is fit on the train fold only (no leakage)
  - evaluate_failure() converts RUL predictions to binary failure labels
  - Per-sample binary predictions are stored for McNemar test
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.cmapss_loader import get_feature_columns, load_train
from src.config import CONFIG
from src.evaluate_failure import evaluate_failure
from src.feature_engineering import add_time_series_features

FAILURE_THRESHOLD = 30


def _binarize(y_true_rul: np.ndarray, y_pred_rul: np.ndarray):
    return (
        (y_true_rul <= FAILURE_THRESHOLD).astype(int),
        (y_pred_rul <= FAILURE_THRESHOLD).astype(int),
    )


def cv_random_forest(
    train_path: str,
    n_folds: int = 5,
    n_estimators: int = 200,
    max_depth: int = None,
    random_state: int = 42,
) -> dict:
    train_df = load_train(train_path)
    _ts = CONFIG.get("time_series_features", {})
    train_df = add_time_series_features(
        train_df,
        windows=tuple(_ts.get("windows", [5, 10, 20])),
        lags=tuple(_ts.get("lags", [1, 5, 10])),
        ewm_span=_ts.get("ewm_span", 10),
    )
    feat_cols = get_feature_columns(train_df)
    groups = train_df["unit_id"].values

    gkf = GroupKFold(n_splits=n_folds)
    precision_scores, recall_scores, roc_auc_scores = [], [], []
    all_y_true_bin, all_y_pred_bin = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups)):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(fold_train[feat_cols])
        y_train = fold_train["RUL"].values

        # Evaluate on ALL cycles of validation engines (both classes present)
        X_val = scaler.transform(fold_val[feat_cols])
        y_val = fold_val["RUL"].values

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        m = evaluate_failure(y_val, y_pred)
        precision_scores.append(m["precision"])
        recall_scores.append(m["recall"])
        roc_auc_scores.append(m["roc_auc"])

        y_true_bin, y_pred_bin = _binarize(y_val, y_pred)
        all_y_true_bin.extend(y_true_bin.tolist())
        all_y_pred_bin.extend(y_pred_bin.tolist())

        print(f"  RF     fold {fold_idx+1}/{n_folds}: precision={m['precision']:.4f}  recall={m['recall']:.4f}  roc_auc={m['roc_auc']:.4f}")

    return {
        "precision": precision_scores,
        "recall": recall_scores,
        "roc_auc": roc_auc_scores,
        "y_true_bin": all_y_true_bin,
        "y_pred_bin": all_y_pred_bin,
    }


def cv_xgboost(
    train_path: str,
    n_folds: int = 5,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> dict:
    train_df = load_train(train_path)
    _ts = CONFIG.get("time_series_features", {})
    train_df = add_time_series_features(
        train_df,
        windows=tuple(_ts.get("windows", [5, 10, 20])),
        lags=tuple(_ts.get("lags", [1, 5, 10])),
        ewm_span=_ts.get("ewm_span", 10),
    )
    feat_cols = get_feature_columns(train_df)
    groups = train_df["unit_id"].values

    gkf = GroupKFold(n_splits=n_folds)
    precision_scores, recall_scores, roc_auc_scores = [], [], []
    all_y_true_bin, all_y_pred_bin = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups)):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(fold_train[feat_cols])
        y_train = fold_train["RUL"].values

        X_val = scaler.transform(fold_val[feat_cols])
        y_val = fold_val["RUL"].values

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_val), 0, None)

        m = evaluate_failure(y_val, y_pred)
        precision_scores.append(m["precision"])
        recall_scores.append(m["recall"])
        roc_auc_scores.append(m["roc_auc"])

        y_true_bin, y_pred_bin = _binarize(y_val, y_pred)
        all_y_true_bin.extend(y_true_bin.tolist())
        all_y_pred_bin.extend(y_pred_bin.tolist())

        print(f"  XGBoost fold {fold_idx+1}/{n_folds}: precision={m['precision']:.4f}  recall={m['recall']:.4f}  roc_auc={m['roc_auc']:.4f}")

    return {
        "precision": precision_scores,
        "recall": recall_scores,
        "roc_auc": roc_auc_scores,
        "y_true_bin": all_y_true_bin,
        "y_pred_bin": all_y_pred_bin,
    }


def cv_lstm(
    train_path: str,
    n_folds: int = 5,
    cv_epochs: int = 20,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 256,
    lr: float = 0.001,
    random_state: int = 42,
) -> dict:
    try:
        import tensorflow as tf
        from tensorflow import keras
    except (ImportError, ModuleNotFoundError) as e:
        print(f"  [LSTM] Skipping -- TensorFlow not available on this platform: {e}")
        return None

    from src.train_dl_rul import WINDOW, _make_sequences

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    train_df = load_train(train_path)
    feat_cols = get_feature_columns(train_df)
    groups = train_df["unit_id"].values

    gkf = GroupKFold(n_splits=n_folds)
    precision_scores, recall_scores, roc_auc_scores = [], [], []
    all_y_true_bin, all_y_pred_bin = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups)):
        fold_train = train_df.iloc[train_idx].copy()
        fold_val = train_df.iloc[val_idx].copy()

        scaler = StandardScaler()
        fold_train[feat_cols] = scaler.fit_transform(fold_train[feat_cols])
        fold_val[feat_cols] = scaler.transform(fold_val[feat_cols])

        # Use all windows from both train and val folds
        X_train_seq, y_train_seq = _make_sequences(fold_train, feat_cols, WINDOW)
        X_val_seq, y_val_seq = _make_sequences(fold_val, feat_cols, WINDOW)

        input_size = X_train_seq.shape[2]
        inputs = keras.Input(shape=(WINDOW, input_size))
        x = inputs
        for i in range(num_layers):
            x = keras.layers.LSTM(hidden_size, return_sequences=(i < num_layers - 1))(x)
            if dropout > 0:
                x = keras.layers.Dropout(dropout)(x)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")

        model.fit(X_train_seq, y_train_seq, epochs=cv_epochs, batch_size=batch_size, verbose=0)

        y_pred = np.clip(model.predict(X_val_seq, verbose=0).flatten(), 0, None)

        m = evaluate_failure(y_val_seq, y_pred)
        precision_scores.append(m["precision"])
        recall_scores.append(m["recall"])
        roc_auc_scores.append(m["roc_auc"])

        y_true_bin, y_pred_bin = _binarize(y_val_seq, y_pred)
        all_y_true_bin.extend(y_true_bin.tolist())
        all_y_pred_bin.extend(y_pred_bin.tolist())

        print(f"  LSTM   fold {fold_idx+1}/{n_folds}: precision={m['precision']:.4f}  recall={m['recall']:.4f}  roc_auc={m['roc_auc']:.4f}")

        keras.backend.clear_session()

    return {
        "precision": precision_scores,
        "recall": recall_scores,
        "roc_auc": roc_auc_scores,
        "y_true_bin": all_y_true_bin,
        "y_pred_bin": all_y_pred_bin,
    }
