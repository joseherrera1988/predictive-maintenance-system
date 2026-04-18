import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.config import CONFIG
from src.evaluate import evaluate_model
from src.model_utils import save_model
from src.tracker import log_experiment


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return torch.sigmoid(self.fc(hn[-1]))


def train(df: pd.DataFrame) -> LSTMClassifier:
    target = CONFIG["features"]["target_column"]
    X = df.drop(columns=[target]).values
    y = df[target].values

    seed = CONFIG["project"]["seed"]
    test_size = CONFIG["data"]["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    dl_cfg = CONFIG["dl_model"]["params"]
    epochs = dl_cfg["epochs"]
    batch_size = dl_cfg["batch_size"]
    lr = dl_cfg["learning_rate"]

    X_tr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    model = LSTMClassifier(
        input_size=X.shape[1],
        hidden_size=dl_cfg["hidden_size"],
        num_layers=dl_cfg["num_layers"],
        dropout=dl_cfg["dropout"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    save_model(model, "lstm")
    return model
