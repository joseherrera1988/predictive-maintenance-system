import json
import pickle
from datetime import datetime
from pathlib import Path
from src.config import CONFIG


def save_model(model, name: str) -> Path:
    models_dir = Path(CONFIG["tracking"]["models_path"])
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = models_dir / f"{name}_{timestamp}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    _update_metadata(name, str(model_path), timestamp)
    print(f"[model_utils] saved model to {model_path}")
    return model_path


def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def _update_metadata(name: str, path: str, timestamp: str) -> None:
    meta_path = Path(CONFIG["tracking"]["metadata_path"])
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        metadata = {"models": []}

    metadata["models"].append({"name": name, "path": path, "saved_at": timestamp})

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
