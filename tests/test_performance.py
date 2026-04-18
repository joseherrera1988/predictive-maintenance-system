import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.evaluate import evaluate_model


def make_dataset(n=500):
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, 10))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_model_f1_above_threshold():
    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    import pandas as pd
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, pd.DataFrame(X_test), pd.Series(y_test))
    assert metrics["f1"] >= 0.70, f"F1 too low: {metrics['f1']:.3f}"


def test_inference_speed(benchmark):
    X, y = make_dataset(n=1000)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    benchmark(model.predict, X_test)
