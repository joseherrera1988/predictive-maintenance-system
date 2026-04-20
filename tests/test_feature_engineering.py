import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import add_time_series_features

WINDOWS = (5, 10, 20)
LAGS = (1, 5, 10)
EWM_SPAN = 10
N_ENGINES = 5
N_CYCLES = 30
SENSOR_COLS = ["s2", "s3", "s4"]


@pytest.fixture
def sample_df():
    rows = []
    for uid in range(1, N_ENGINES + 1):
        for cycle in range(1, N_CYCLES + 1):
            row = {"unit_id": uid, "cycle": cycle, "RUL": N_CYCLES - cycle}
            for col in SENSOR_COLS:
                row[col] = float(uid * 100 + cycle)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def enriched_df(sample_df):
    return add_time_series_features(
        sample_df, sensor_cols=SENSOR_COLS, windows=WINDOWS, lags=LAGS, ewm_span=EWM_SPAN
    )


def test_no_rows_dropped(sample_df, enriched_df):
    assert len(enriched_df) == len(sample_df)


def test_new_columns_added(enriched_df):
    for col in SENSOR_COLS:
        for w in WINDOWS:
            assert f"{col}_mean_{w}" in enriched_df.columns
            assert f"{col}_std_{w}" in enriched_df.columns
        for l in LAGS:
            assert f"{col}_lag_{l}" in enriched_df.columns
        assert f"{col}_ewm_{EWM_SPAN}" in enriched_df.columns
    assert "cycle_pct" in enriched_df.columns


def test_no_nan_in_output(enriched_df):
    assert enriched_df.isnull().sum().sum() == 0


def test_rolling_is_per_engine(sample_df):
    # Engine 1 values are ~100+cycle, engine 2 are ~200+cycle — rolling mean must not bleed across
    df = add_time_series_features(
        sample_df, sensor_cols=["s2"], windows=(5,), lags=(), ewm_span=10
    )
    eng1_mean = df[df["unit_id"] == 1]["s2_mean_5"].max()
    eng2_mean = df[df["unit_id"] == 2]["s2_mean_5"].min()
    assert eng1_mean < eng2_mean


def test_cycle_pct_range(enriched_df):
    assert enriched_df["cycle_pct"].between(0, 1, inclusive="right").all()


def test_feature_count(sample_df):
    df = add_time_series_features(
        sample_df, sensor_cols=SENSOR_COLS, windows=WINDOWS, lags=LAGS, ewm_span=EWM_SPAN
    )
    from src.cmapss_loader import get_feature_columns
    # original SENSOR_COLS + new features + cycle_pct
    # per sensor: len(WINDOWS)*2 + len(LAGS) + 1 (ewm) new cols; plus cycle_pct
    expected_new = len(SENSOR_COLS) * (len(WINDOWS) * 2 + len(LAGS) + 1) + 1
    original_non_meta = len(SENSOR_COLS)
    # count cols not in original sample_df
    original_cols = set(sample_df.columns)
    new_cols = [c for c in df.columns if c not in original_cols]
    assert len(new_cols) == expected_new


def test_original_columns_unchanged(sample_df, enriched_df):
    for col in SENSOR_COLS:
        pd.testing.assert_series_equal(
            sample_df[col].reset_index(drop=True),
            enriched_df[col].reset_index(drop=True),
        )


def test_lag_bfill_no_nan(sample_df):
    df = add_time_series_features(
        sample_df, sensor_cols=["s2"], windows=(), lags=(1,), ewm_span=10
    )
    # lag_1 of first cycle should equal first cycle's own value (bfill applied)
    first_cycles = df[df["cycle"] == 1]
    assert (first_cycles["s2_lag_1"] == first_cycles["s2"]).all()
