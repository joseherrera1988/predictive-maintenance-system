import csv
import uuid
from datetime import datetime
from pathlib import Path
from src.config import CONFIG


def log_experiment(model_type: str, params: dict, metrics: dict, notes: str = "") -> str:
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().isoformat()
    logs_path = Path(CONFIG["tracking"]["experiments_path"])
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    write_header = not logs_path.exists()
    with open(logs_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "timestamp", "model_type", "params",
                        "accuracy", "precision", "recall", "f1", "roc_auc", "notes"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            "run_id": run_id,
            "timestamp": timestamp,
            "model_type": model_type,
            "params": str(params),
            **metrics,
            "notes": notes,
        })

    print(f"[tracker] run {run_id} logged — {metrics}")
    return run_id
