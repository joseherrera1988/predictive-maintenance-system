import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from src.preprocessing import preprocess
from src.evaluate import evaluate_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def make_raw_df(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "temperature": rng.normal(75, 10, n),
        "vibration": rng.normal(0.5, 0.1, n),
        "pressure": rng.normal(100, 5, n),
        "failure": rng.integers(0, 2, n),
    })


def test_full_pipeline_end_to_end():
    df = make_raw_df()
    df_clean = preprocess(df)

    target = "failure"
    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["accuracy"] > 0.0
