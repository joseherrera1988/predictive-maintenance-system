# API Reference

All public functions and classes across the `src/` package.

---

## `src/config.py`

### `load_config(config_path="configs/config.yaml") → dict`
Reads and parses the YAML config file. Called once at import time; result stored in module-level `CONFIG`.

**Module-level**

| Symbol | Type | Description |
|--------|------|-------------|
| `CONFIG` | `dict` | Loaded config, imported by all other modules |

---

## `src/data_pipeline.py`

### `load_raw_data(filename: str) → pd.DataFrame`
Reads `data/raw/<filename>` using `CONFIG["data"]["raw_path"]`.

### `save_processed_data(df: pd.DataFrame, filename: str) → None`
Writes cleaned DataFrame to `data/processed/<filename>`.

### `run_pipeline(filename: str) → pd.DataFrame`
Convenience wrapper: `load_raw_data` → `preprocess` → `save_processed_data`. Returns the cleaned DataFrame.

---

## `src/preprocessing.py`

### `drop_columns(df: pd.DataFrame) → pd.DataFrame`
Drops columns listed in `CONFIG["features"]["drop_columns"]`. Silently ignores missing column names.

### `encode_categoricals(df: pd.DataFrame) → pd.DataFrame`
Label-encodes columns in `CONFIG["features"]["categorical_columns"]` using `pd.Categorical`.

### `scale_numericals(df: pd.DataFrame) → pd.DataFrame`
Applies `StandardScaler` to columns in `CONFIG["features"]["numerical_columns"]`.

### `preprocess(df: pd.DataFrame) → pd.DataFrame`
Full preprocessing chain: `drop_columns` → `encode_categoricals` → `scale_numericals` → `dropna`.

---

## `src/train_ml.py`

### `train(df: pd.DataFrame) → RandomForestClassifier`
Splits data, trains a `RandomForestClassifier` with params from `CONFIG["ml_model"]["params"]`, evaluates, saves, and logs the run.

---

## `src/train_dl.py`

### `class LSTMClassifier(nn.Module)`
PyTorch LSTM classifier with a single linear output and sigmoid activation.

| Method | Description |
|--------|-------------|
| `__init__(input_size, hidden_size, num_layers, dropout)` | Builds LSTM + FC head |
| `forward(x) → Tensor` | Returns sigmoid probability per sample |

### `train(df: pd.DataFrame) → LSTMClassifier`
Splits data, builds a `DataLoader`, trains with Adam + BCELoss for configured epochs, and saves the model.

---

## `src/evaluate.py`

### `evaluate_model(model, X_test, y_test) → dict`
Runs `model.predict()` and (if available) `model.predict_proba()` and returns:

| Key | Metric |
|-----|--------|
| `accuracy` | `sklearn.metrics.accuracy_score` |
| `precision` | `sklearn.metrics.precision_score` |
| `recall` | `sklearn.metrics.recall_score` |
| `f1` | `sklearn.metrics.f1_score` |
| `roc_auc` | `sklearn.metrics.roc_auc_score` (NaN if probabilities unavailable) |

---

## `src/tracker.py`

### `log_experiment(model_type: str, params: dict, metrics: dict, notes: str = "") → str`
Appends one row to `experiments/logs.csv` (path from `CONFIG["tracking"]["experiments_path"]`).
Returns the 8-character `run_id`.

**CSV columns:** `run_id`, `timestamp`, `model_type`, `params`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `notes`

---

## `src/model_utils.py`

### `save_model(model, name: str) → Path`
Pickles the model to `models/saved_models/<name>_<TIMESTAMP>.pkl` and updates `metadata.json`. Returns the saved path.

### `load_model(model_path: str) → object`
Loads and returns a pickled model from the given path.

### `_update_metadata(name: str, path: str, timestamp: str) → None`
Internal. Appends `{name, path, saved_at}` to `models/metadata.json`.
