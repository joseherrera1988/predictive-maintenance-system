# Usage Guide

## Prerequisites

```bash
pip install -r requirements.txt
```

Place your sensor CSV file in `data/raw/` before running training.

---

## Train a Model

```bash
# Train Random Forest only (default)
python main.py --data sensors.csv --model ml

# Train LSTM only
python main.py --data sensors.csv --model dl

# Train both sequentially
python main.py --data sensors.csv --model both
```

### What happens

1. `data/raw/sensors.csv` is loaded
2. Preprocessing runs: drop configured columns → encode categoricals → scale numericals → drop nulls
3. Cleaned DataFrame is saved to `data/processed/sensors.csv`
4. Selected model(s) are trained on an 80/20 train-test split (seed=42)
5. Evaluation metrics are computed (accuracy, precision, recall, F1, ROC-AUC)
6. Model is pickled to `models/saved_models/`
7. Run is appended to `experiments/logs.csv`

---

## Configure the Pipeline

Edit `configs/config.yaml` to control all behaviour without touching code:

```yaml
features:
  target_column: failure          # column to predict
  drop_columns: [id, timestamp]   # columns to remove before training
  categorical_columns: [machine]  # columns to label-encode
  numerical_columns: [temp, vib]  # columns to StandardScale

ml_model:
  params:
    n_estimators: 100
    max_depth: 10

dl_model:
  params:
    hidden_size: 64
    num_layers: 2
    dropout: 0.2
    epochs: 50
    batch_size: 32
    learning_rate: 0.001
```

---

## Launch the Dashboard

```bash
streamlit run dashboards/streamlit_app.py
```

Opens at `http://localhost:8501`. Three sections:

| Section | Source |
|---------|--------|
| Experiment Logs | `experiments/logs.csv` |
| Saved Models | `models/metadata.json` |
| Run Prediction | CSV file upload (preview only — wire up a model to score) |

---

## Run Tests

```bash
# All tests except performance benchmarks
pytest tests/ -v --ignore=tests/test_performance.py

# Performance benchmarks (requires pytest-benchmark)
pytest tests/test_performance.py -v
```

| Test file | What it covers |
|-----------|---------------|
| `test_preprocessing.py` | drop, null removal, output type |
| `test_model.py` | RandomForest trains and returns correct metric keys |
| `test_integration.py` | full preprocess → train → evaluate pipeline |
| `test_performance.py` | F1 threshold ≥ 0.70, inference speed benchmark |
