import pandas as pd
from pathlib import Path
from src.config import CONFIG


def load_raw_data(filename: str) -> pd.DataFrame:
    raw_path = Path(CONFIG["data"]["raw_path"]) / filename
    return pd.read_csv(raw_path)


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    processed_path = Path(CONFIG["data"]["processed_path"]) / filename
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_path, index=False)


def run_pipeline(filename: str) -> pd.DataFrame:
    from src.preprocessing import preprocess

    df = load_raw_data(filename)
    df = preprocess(df)
    save_processed_data(df, filename)
    return df
