"""
Group K-Fold cross-validation wrappers for RF, XGBoost, and LSTM.

Splits engines (unit_id) into k folds; trains on k-1 groups, validates on 1.
Preserves temporal structure within each engine's cycle history.

Per fold:
  - StandardScaler is fit on the train fold only (no leakage)
  - Validation uses the last cycle per engine (RF/XGBoost) or last window (LSTM)
  - evaluate_failure() converts RUL predictions to binary failure labels
  - Per-sample binary predictions are stored for McNemar test
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.cmapss_loader import get_feature_columns, load_train
from src.evaluate_failure import evaluate_failure
from src.train_dl_rul import WINDOW, _last_window_per_engine, _make_sequences

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
    feat_cols = get_feature_columns(train_df)
    groups = train_df["unit_id"].values

    gkf = GroupKFold(n_splits=n_folds)
    precision_scores, roc_auc_scores = [], []
    all_y_true_bin, all_y_pred_bin = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups)):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(fold_train[feat_cols])
        y_train = fold_train["RUL"].values

        fold_val_last = fold_val.groupby("unit_id").last().reset_index()
        X_val = scaler.transform(fold_val_last[feat_cols])
        y_val = fold_val_last["RUL"].values

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
        roc_auc_scores.append(m["roc_auc"])

        y_true_bin, y_pred_bin = _binarize(y_val, y_pred)
        all_y_true_bin.extend(y_true_bin.tolist())
        all_y_pred_bin.extend(y_pred_bin.tolist())

        print(f"  RF     fold {fold_idx+1}/{n_folds}: precision={m['precision']:.4f}  roc_auc={m['roc_auc']:.4f}")

    return {
        "precision": precision_scores,
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
    feat_cols = get_feature_columns(train_df)
    groups = train_df["unit_id"].values

    gkf = GroupKFold(n_splits=n_folds)
    precision_scores, roc_auc_scores = [], []
    all_y_true_bin, all_y_pred_bin = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups)):
        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(fold_train[feat_cols])
        y_train = fold_train["RUL"].values

        fold_val_last = fold_val.groupby("unit_id").last().reset_index()
        X_val = scaler.transform(fold_val_last[feat_cols])
        y_val = fold_val_last["RUL"].values

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
        roc_auc_scores.append(m["roc_auc"])

        y_true_bin, y_pred_bin = _binarize(y_val, y_pred)
        all_y_true_bin.extend(y_true_bin.tolist())
        all_y_pred_bin.extend(y_pred_bin.tolist())

        print(f"  XGBoost fold {fold_idx+1}/{n_folds}: precision={m['precision']:.4f}  roc_auc={m['roc_auc']:.4f}")

    return {
        "precision": precision_scores,
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
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    train_df = load_train(train_path)
    feat_cols = get_feature_columns(train_df)
    groups = train_df["unit_id"].values

    gkf = GroupKFold(n_splits=n_folds)
    precision_scores, roc_auc_scores = [], []
    all_y_true_bin, all_y_pred_bin = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups)):
        fold_train = train_df.iloc[train_idx].copy()
        fold_val = train_df.iloc[val_idx].copy()

        scaler = StandardScaler()
        fold_train[feat_cols] = scaler.fit_transform(fold_train[feat_cols])

        X_train_seq, y_train_seq = _make_sequences(fold_train, feat_cols, WINDOW)

        # Last window per validation engine; y = last RUL per engine
        X_val_seq = _last_window_per_engine(fold_val, feat_cols, WINDOW, scaler)
        y_val = fold_val.groupby("unit_id")["RUL"].last().values.astype(np.float32)

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

        m = evaluate_failure(y_val, y_pred)
        precision_scores.append(m["precision"])
        roc_auc_scores.append(m["roc_auc"])

        y_true_bin, y_pred_bin = _binarize(y_val, y_pred)
        all_y_true_bin.extend(y_true_bin.tolist())
        all_y_pred_bin.extend(y_pred_bin.tolist())

        print(f"  LSTM   fold {fold_idx+1}/{n_folds}: precision={m['precision']:.4f}  roc_auc={m['roc_auc']:.4f}")

        keras.backend.clear_session()

    return {
        "precision": precision_scores,
        "roc_auc": roc_auc_scores,
        "y_true_bin": all_y_true_bin,
        "y_pred_bin": all_y_pred_bin,
    }
