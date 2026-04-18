import pandas as pd
import pytest
from src.preprocessing import drop_columns, encode_categoricals, preprocess


def sample_df():
    return pd.DataFrame({
        "temperature": [70.0, 80.0, 90.0, None],
        "vibration": [0.1, 0.2, 0.3, 0.4],
        "failure": [0, 0, 1, 0],
    })


def test_drop_columns_removes_nothing_by_default():
    df = sample_df()
    result = drop_columns(df)
    assert set(result.columns) == set(df.columns)


def test_preprocess_removes_nulls():
    df = sample_df()
    result = preprocess(df)
    assert result.isnull().sum().sum() == 0


def test_preprocess_returns_dataframe():
    df = sample_df().dropna()
    result = preprocess(df)
    assert isinstance(result, pd.DataFrame)
