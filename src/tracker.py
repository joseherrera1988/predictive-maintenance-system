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

    base_fields = ["run_id", "timestamp", "model_type", "params"]
    metric_fields = list(metrics.keys())
    fieldnames = base_fields + metric_fields + ["notes"]

    row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_type": model_type,
        "params": str(params),
        **metrics,
        "notes": notes,
    }

    # If the file already exists, read existing headers and merge
    if logs_path.exists() and logs_path.stat().st_size > 0:
        with open(logs_path, "r", newline="") as f:
            existing_fields = csv.DictReader(f).fieldnames or []
        # Add any new metric columns not previously seen
        for field in fieldnames:
            if field not in existing_fields:
                existing_fields.append(field)
        fieldnames = existing_fields

    write_header = not logs_path.exists() or logs_path.stat().st_size == 0
    with open(logs_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[tracker] run {run_id} logged — {metrics}")
    return run_id
