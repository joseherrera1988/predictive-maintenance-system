import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import CONFIG


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = CONFIG["features"].get("drop_columns", [])
    return df.drop(columns=cols, errors="ignore")


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = CONFIG["features"].get("categorical_columns", [])
    for col in cat_cols:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes
    return df


def scale_numericals(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = CONFIG["features"].get("numerical_columns", [])
    existing = [c for c in num_cols if c in df.columns]
    if existing:
        scaler = StandardScaler()
        df[existing] = scaler.fit_transform(df[existing])
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_columns(df)
    df = encode_categoricals(df)
    df = scale_numericals(df)
    df = df.dropna()
    return df
