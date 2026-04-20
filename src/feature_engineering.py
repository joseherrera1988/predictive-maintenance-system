"""
Time-series feature engineering for RF and XGBoost.

All features are computed per engine (groupby unit_id) to prevent
cross-engine contamination. Safe to call before GroupKFold splitting
because rolling/lag operations never cross engine boundaries.
"""

import pandas as pd

from src.cmapss_loader import get_feature_columns


def add_time_series_features(
    df: pd.DataFrame,
    sensor_cols: list = None,
    windows: tuple = (5, 10, 20),
    lags: tuple = (1, 5, 10),
    ewm_span: int = 10,
) -> pd.DataFrame:
    df = df.copy()
    if sensor_cols is None:
        sensor_cols = get_feature_columns(df)

    new_cols = {}

    for col in sensor_cols:
        grp = df.groupby("unit_id")[col]

        for w in windows:
            new_cols[f"{col}_mean_{w}"] = grp.transform(
                lambda x, w=w: x.rolling(w, min_periods=1).mean()
            )
            new_cols[f"{col}_std_{w}"] = grp.transform(
                lambda x, w=w: x.rolling(w, min_periods=2).std().bfill().fillna(0)
            )

        for l in lags:
            new_cols[f"{col}_lag_{l}"] = grp.transform(
                lambda x, l=l: x.shift(l).bfill()
            )

        new_cols[f"{col}_ewm_{ewm_span}"] = grp.transform(
            lambda x: x.ewm(span=ewm_span, adjust=False).mean()
        )

    new_cols["cycle_pct"] = df["cycle"] / df.groupby("unit_id")["cycle"].transform("max")

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
