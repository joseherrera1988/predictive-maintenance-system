# Predictive Maintenance AI

An end-to-end ML/DL pipeline for predicting industrial equipment failures using sensor time-series data. Compares classical ML (Random Forest) and deep learning (LSTM) approaches with full experiment tracking and a Streamlit monitoring dashboard.

## Core Components

| Module | File | Responsibility |
|--------|------|----------------|
| Config loader | `src/config.py` | Parses `configs/config.yaml` into `CONFIG` dict |
| Data pipeline | `src/data_pipeline.py` | Load raw CSV → preprocess → save to processed |
| Preprocessing | `src/preprocessing.py` | Drop, encode, scale, drop nulls |
| ML training | `src/train_ml.py` | RandomForest train, evaluate, save, log |
| DL training | `src/train_dl.py` | LSTM train via PyTorch, save |
| Evaluation | `src/evaluate.py` | accuracy, precision, recall, F1, ROC-AUC |
| Tracker | `src/tracker.py` | Appends run row to `experiments/logs.csv` |
| Model utils | `src/model_utils.py` | Pickle save/load + `metadata.json` registry |
| Dashboard | `dashboards/streamlit_app.py` | Logs viewer, model table, prediction uploader |
| Entry point | `main.py` | CLI: `--data`, `--model` |

## Quick Links

- [Usage Guide](usage.md) — how to run training, the dashboard, and tests
- [API Reference](api.md) — module-level function signatures
- [Architecture Diagrams](architecture.md) — Mermaid flowcharts
