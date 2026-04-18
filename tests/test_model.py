import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split


def make_dataset():
    import numpy as np
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = (X[:, 0] > 0).astype(int)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]), pd.Series(y)


def test_random_forest_trains_and_evaluates():
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_returns_all_keys():
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert key in metrics
