import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.config import CONFIG
from src.evaluate import evaluate_model
from src.model_utils import save_model
from src.tracker import log_experiment


def train(df: pd.DataFrame) -> RandomForestClassifier:
    target = CONFIG["features"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    test_size = CONFIG["data"]["test_size"]
    seed = CONFIG["project"]["seed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    params = CONFIG["ml_model"]["params"]
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, "random_forest")
    log_experiment("random_forest", params, metrics)

    return model
