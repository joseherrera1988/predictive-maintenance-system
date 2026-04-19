"""
LSTM regressor for CMAPSS RUL prediction using TensorFlow/Keras.

Each sample is a window of W consecutive cycles for one engine.
The target is the RUL at the last timestep of that window.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.cmapss_loader import load_train, load_test, get_feature_columns, _read_txt, DROP_SENSORS
from src.evaluate_regression import evaluate_regression
from src.evaluate_failure import evaluate_failure, print_failure_metrics
from src.model_utils import save_model
from src.tracker import log_experiment

WINDOW = 30  # cycles per sequence


# ── Windowing ──────────────────────────────────────────────────────────────────
def _make_sequences(
    df,
    feat_cols: list[str],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slides a window of `window` cycles over each engine's history.
    Pads with zeros at the start when fewer than `window` cycles exist.
    Returns X of shape (N, window, features) and y of shape (N,).
    """
    import pandas as pd
    X_list, y_list = [], []
    for _, group in df.groupby("unit_id"):
        feats = group[feat_cols].values
        targets = group["RUL"].values
        n = len(group)
        for i in range(n):
            end = i + 1
            start = max(0, end - window)
            seq = feats[start:end]
            pad_len = window - len(seq)
            if pad_len > 0:
                seq = np.vstack([np.zeros((pad_len, seq.shape[1])), seq])
            X_list.append(seq)
            y_list.append(targets[i])
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def _last_window_per_engine(
    df,
    feat_cols: list[str],
    window: int,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    For test data: one window per engine using the last `window` cycles.
    Returns X of shape (n_engines, window, features).
    """
    X_list = []
    for _, group in df.groupby("unit_id"):
        feats = group[feat_cols].values
        feats = scaler.transform(feats)
        seq = feats[-window:] if len(feats) >= window else feats
        pad_len = window - len(seq)
        if pad_len > 0:
            seq = np.vstack([np.zeros((pad_len, seq.shape[1])), seq])
        X_list.append(seq)
    return np.array(X_list, dtype=np.float32)


# ── Training ───────────────────────────────────────────────────────────────────
def train(
    train_path: str = "data/raw/train_FD001.txt",
    test_path: str = "data/raw/test_FD001.txt",
    rul_path: str = "data/raw/RUL_FD001.txt",
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.001,
    random_state: int = 42,
):
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    # ── Load ───────────────────────────────────────────────────────────────────
    train_df = load_train(train_path)
    test_df = load_test(test_path, rul_path)
    feat_cols = get_feature_columns(train_df)

    # ── Scale (fit on training features only) ──────────────────────────────────
    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])

    # ── Build sequences ────────────────────────────────────────────────────────
    X_train, y_train = _make_sequences(train_df, feat_cols, WINDOW)

    raw_test = _read_txt(test_path).drop(columns=DROP_SENSORS)
    X_test = _last_window_per_engine(raw_test, feat_cols, WINDOW, scaler)
    y_test = test_df["RUL"].values.astype(np.float32)

    # ── Build model ────────────────────────────────────────────────────────────
    input_size = X_train.shape[2]
    inputs = keras.Input(shape=(WINDOW, input_size))
    x = inputs
    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        x = keras.layers.LSTM(hidden_size, return_sequences=return_sequences)(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
    )
    model.summary()

    # ── Train ──────────────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred = model.predict(X_test, verbose=0).flatten()
    y_pred = np.clip(y_pred, 0, None)
    metrics = evaluate_regression(y_test, y_pred)

    print("\n[DL] LSTM Regressor (Keras) — FD001 Results")
    print(f"  RMSE       : {metrics['rmse']}")
    print(f"  MAE        : {metrics['mae']}")
    print(f"  R²         : {metrics['r2']}")
    print(f"  NASA Score : {metrics['nasa_score']}")

    cls_metrics = evaluate_failure(y_test, y_pred)
    print_failure_metrics(cls_metrics)

    # ── Save & log ─────────────────────────────────────────────────────────────
    params = {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "window": WINDOW,
    }
    save_model(model, "lstm_rul_fd001")
    log_experiment("lstm_regressor_keras", params, metrics, notes="CMAPSS FD001")

    return model
