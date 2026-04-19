# Predictive Maintenance System for Industrial Equipment

## Overview
This project builds an end-to-end machine learning system to predict equipment failure using time-series sensor data. It compares classical ML models and deep learning approaches to evaluate tradeoffs in performance and interpretability.

## Features
- Time-series feature engineering (rolling mean, variance, trends)
- Classical ML models (Random Forest, XGBoost)
- LSTM-based deep learning model
- Model evaluation using ROC-AUC, precision, recall
- Streamlit dashboard for real-time monitoring

## Tech Stack
Python, Pandas, Scikit-learn, PyTorch, Streamlit

## Project Structure

```
predictive-maintenance-ai/
├── configs/config.yaml       # Central configuration
├── data/raw/                 # Raw input CSVs
├── data/processed/           # Preprocessed data
├── models/saved_models/      # Serialized trained models
├── models/metadata.json      # Model registry
├── experiments/logs.csv      # Experiment tracking
├── src/                      # Core source code
├── dashboards/streamlit_app.py
├── tests/                    # Unit, integration, performance tests
└── main.py                   # CLI entry point
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Place your CSV in data/raw/, then run:
python main.py --data your_file.csv --model both

# Launch dashboard
streamlit run dashboards/streamlit_app.py
```

## Training

| Flag | Options | Description |
|------|---------|-------------|
| `--train` | filename | Path to train txt file |
| `--test` | filename | Path to test txt file |
| `--rul` | filename | Path to RUL txt file |
| `--model` | `ml`, `dl`, `xgb`, `all` | `ml`=Random Forest, `dl`=LSTM, `xgb`=XGBoost, `all`=run all three |

## Results

Evaluated on NASA CMAPSS FD001–FD004. Failure threshold: RUL ≤ 30 cycles.

### Recall
| Dataset | RF     | XGBoost | LSTM   |
|---------|--------|---------|--------|
| FD001   | 0.68   | 0.72    | **0.80** |
| FD002   | 0.90   | 0.93    | **0.97** |
| FD003   | 0.75   | 0.65    | **0.95** |
| FD004   | 0.74   | **0.75** | 0.66   |

### Precision
| Dataset | RF     | XGBoost | LSTM   |
|---------|--------|---------|--------|
| FD001   | 0.944  | **0.947** | 0.909 |
| FD002   | 0.965  | 0.934   | **0.967** |
| FD003   | 0.938  | 0.929   | **0.950** |
| FD004   | **0.929** | 0.909 | 0.854  |

### ROC-AUC
| Dataset | RF     | XGBoost | LSTM   |
|---------|--------|---------|--------|
| FD001   | 0.974  | 0.970   | **0.990** |
| FD002   | 0.990  | 0.988   | **0.998** |
| FD003   | 0.989  | 0.985   | **0.995** |
| FD004   | **0.979** | **0.979** | 0.977 |

## Testing

```bash
pytest tests/ -v
```

## Configuration

Edit `configs/config.yaml` to adjust model hyperparameters, data splits, feature columns, and file paths.
