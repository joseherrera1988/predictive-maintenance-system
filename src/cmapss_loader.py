"""
Loader for the NASA CMAPSS FD001 dataset.

Column layout (26 cols, no header):
  unit_id, cycle,
  setting_1, setting_2, setting_3,
  s1 .. s21
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Column names ───────────────────────────────────────────────────────────────
COLUMNS = (
    ["unit_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance on FD001 — drop them
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

# Cap RUL at this value (piecewise-linear assumption: engine is "as good as new" above it)
RUL_CAP = 125


def _read_txt(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)


def load_train(path: str = "data/raw/train_FD001.txt") -> pd.DataFrame:
    """
    Loads training data and computes capped RUL for every row.
    RUL = max_cycle_in_unit - current_cycle, capped at RUL_CAP.
    """
    df = _read_txt(path)
    max_cycles = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycles, on="unit_id")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=RUL_CAP)
    df = df.drop(columns=["max_cycle"] + DROP_SENSORS)
    return df.reset_index(drop=True)


def load_test(
    test_path: str = "data/raw/test_FD001.txt",
    rul_path: str = "data/raw/RUL_FD001.txt",
) -> pd.DataFrame:
    """
    Loads test data (last observed cycle per engine) and attaches ground-truth RUL.
    Returns one row per engine (the last cycle), with RUL capped at RUL_CAP.
    """
    df = _read_txt(test_path)
    rul = pd.read_csv(rul_path, header=None, names=["RUL"])
    rul["unit_id"] = rul.index + 1
    rul["RUL"] = rul["RUL"].clip(upper=RUL_CAP)

    # Keep only the last observed cycle per engine
    last_cycles = df.groupby("unit_id").last().reset_index()
    last_cycles = last_cycles.drop(columns=DROP_SENSORS)
    last_cycles = last_cycles.merge(rul[["unit_id", "RUL"]], on="unit_id")
    return last_cycles.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Returns sensor + setting columns (excludes unit_id, cycle, RUL)."""
    exclude = {"unit_id", "cycle", "RUL"}
    return [c for c in df.columns if c not in exclude]
