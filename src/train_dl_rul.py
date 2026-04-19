"""
LSTM regressor for CMAPSS RUL prediction using a sliding window approach.

Each sample is a window of W consecutive cycles for one engine.
The target is the RUL at the last timestep of that window.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.cmapss_loader import load_train, load_test, get_feature_columns
from src.evaluate_regression import evaluate_regression
from src.model_utils import save_model
from src.tracker import log_experiment


WINDOW = 30  # cycles per sequence


# ── Model ──────────────────────────────────────────────────────────────────────
class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1]).squeeze(-1)


# ── Windowing ──────────────────────────────────────────────────────────────────
def _make_sequences(
    df: pd.DataFrame,
    feat_cols: list[str],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slides a window of `window` cycles over each engine's history.
    Pads with zeros at the start when fewer than `window` cycles exist.
    Returns X of shape (N, window, features) and y of shape (N,).
    """
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
    df: pd.DataFrame,
    feat_cols: list[str],
    window: int,
    scaler: StandardScaler,
) -> np.ndarray:
    """
    For test data: one window per engine (the last `window` cycles).
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
) -> LSTMRegressor:

    torch.manual_seed(random_state)

    # ── Load ───────────────────────────────────────────────────────────────────
    train_df = load_train(train_path)
    test_df = load_test(test_path, rul_path)
    feat_cols = get_feature_columns(train_df)

    # ── Scale (fit on training features only) ──────────────────────────────────
    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])

    # ── Build sequences ────────────────────────────────────────────────────────
    X_train, y_train = _make_sequences(train_df, feat_cols, WINDOW)

    # Test: load raw, scale with training scaler, take last window per engine
    from src.cmapss_loader import _read_txt, DROP_SENSORS
    raw_test = _read_txt(test_path).drop(columns=DROP_SENSORS)
    X_test = _last_window_per_engine(raw_test, feat_cols, WINDOW, scaler)
    y_test = test_df["RUL"].values.astype(np.float32)

    # ── DataLoader ─────────────────────────────────────────────────────────────
    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── Model ──────────────────────────────────────────────────────────────────
    input_size = X_train.shape[2]
    model = LSTMRegressor(input_size, hidden_size, num_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # ── Train loop ─────────────────────────────────────────────────────────────
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={epoch_loss/len(X_train):.4f}")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X_test)).numpy()

    y_pred = np.clip(y_pred, 0, None)
    metrics = evaluate_regression(y_test, y_pred)

    print("\n[DL] LSTM Regressor — FD001 Results")
    print(f"  RMSE       : {metrics['rmse']}")
    print(f"  MAE        : {metrics['mae']}")
    print(f"  R²         : {metrics['r2']}")
    print(f"  NASA Score : {metrics['nasa_score']}")

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
    log_experiment("lstm_regressor", params, metrics, notes="CMAPSS FD001")

    return model
