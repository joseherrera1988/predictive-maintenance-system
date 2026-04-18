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
| `--data` | filename | CSV in `data/raw/` |
| `--model` | `ml`, `dl`, `both` | Model type to train |

## Results

| Model         | ROC-AUC | Precision | Recall |
|---------------|---------|-----------|--------|
| Random Forest | 0.XX    | 0.XX      | 0.XX   |
| XGBoost       | 0.XX    | 0.XX      | 0.XX   |
| LSTM          | 0.XX    | 0.XX      | 0.XX   |

## Testing

```bash
pytest tests/ -v
```

## Configuration

Edit `configs/config.yaml` to adjust model hyperparameters, data splits, feature columns, and file paths.
