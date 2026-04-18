# Predictive Maintenance AI

An end-to-end ML/DL pipeline for predicting equipment failures before they occur.

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

## Testing

```bash
pytest tests/ -v
```

## Configuration

Edit `configs/config.yaml` to adjust model hyperparameters, data splits, feature columns, and file paths.
