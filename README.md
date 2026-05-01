# Predictive Maintenance System for Industrial Equipment

## What This Is

This project builds and evaluates a machine learning system for predicting equipment failure using time-series sensor data from NASA’s CMAPSS turbofan engine datasets (FD001–FD004).

The project addresses a key challenge in applied machine learning:

> **Do complex deep learning models actually outperform simpler methods under real-world conditions?**

The system evaluates the following models:
- Random Forest (RF)
- XGBoost
- LSTM (sequence model)

These models are assessed using a **nested cross-validation and statistical testing framework** that simulates production-like conditions.

## Why This Matters

Predictive maintenance operates in a high-stakes environment:

- Missing a failure (false negative) can cause catastrophic system damage.  
- False alarms (false positives) increase maintenance costs.  
- Systems must generalize across changing operating conditions. 

The key question is not just accuracy. We should also ask **which model is reliable enough to deploy.**

This project evaluates:
- Whether sequence models (LSTM) are necessary.  
- Whether feature engineering can approximate temporal learning.  
- How models behave under distribution shift and multi-condition data.

---

## What This Repository Contains

- A full **time-series feature engineering pipeline** (188 features: rolling stats, lags, EWM, cycle %)
- Classical ML models (RF, XGBoost) with **Optuna-tuned nested cross-validation**
- An LSTM sequence model trained on sliding-window sensor data
- Group K-Fold cross-validation by engine (prevents leakage)
- A statistical testing suite:
  - Friedman test  
  - Pairwise Wilcoxon (Bonferroni corrected)  
  - Paired t-test  
  - McNemar’s test  
- Experiment tracking:
  - Per-run logs  
  - Consolidated benchmark tables  
  - Statistical reports  
- CLI pipelines for reproducible training, evaluation, and comparison
- (In progress) Streamlit dashboard for monitoring predictions

## Why Evaluation-First

The hardest part of applied ML is not building models; instead, it is determining whether improvements are real.

This repository is built on the principle that a rigorous evaluation framework is more valuable than a complex model without measurement.

Key design choices:

- **Nested cross-validation** prevents overfitting during tuning  
- **Group splits by engine** eliminate data leakage  
- **Multiple metrics (recall, precision, ROC-AUC)** capture different failure modes  
- **Statistical tests** distinguish real improvements from noise  

A single metric (e.g., ROC-AUC) can hide critical issues.  
For example:
- A model may achieve high ROC-AUC but miss failures (low recall)
- A model may have high average performance but high variance across conditions

This project explicitly measures these tradeoffs.
## Project Structure

```
predictive-maintenance-ai/
├── configs/config.yaml              # Central configuration
├── data/raw/                        # Raw CMAPSS txt files
├── data/processed/                  # Preprocessed data
├── models/saved_models/             # Serialized trained models
├── models/metadata.json             # Model registry
├── experiments/logs.csv             # Per-run experiment tracking
├── experiments/results_summary.md   # Consolidated benchmark tables
├── experiments/statistical_report.txt       # Significance tests across FD001–FD004
├── experiments/tuned_results_FD001_FD002.txt # Nested-CV + Optuna results
├── src/feature_engineering.py       # 188-feature time-series pipeline
├── src/cross_validate.py            # Group K-Fold CV
├── src/nested_cv.py                 # Nested CV + Optuna tuning
├── src/statistical_tests.py         # Friedman / Wilcoxon / t-test / McNemar
├── src/                             # Loaders, training, evaluation, etc.
├── dashboards/streamlit_app.py
├── tests/                           # Unit, integration, performance tests
├── main_cmapss.py                   # CMAPSS CLI entry point
├── run_statistical_analysis.py      # Full benchmark + significance CLI
└── main.py                          # Generic CLI entry point
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run the CMAPSS benchmark (single dataset)
python main_cmapss.py --train train_FD001.txt --test test_FD001.txt --rul RUL_FD001.txt --model all

# Run nested CV + statistical significance across all four datasets
python run_statistical_analysis.py \
    --dataset all --folds 5 --n-trials 50 --cv-epochs 50 \
    --tuned --output experiments/tuned_statistical_report.txt

# Launch dashboard
streamlit run dashboards/streamlit_app.py
```

## Training

| Flag      | Options                  | Description                                                       |
| --------- | ------------------------ | ----------------------------------------------------------------- |
| `--train` | filename                 | Path to train txt file                                            |
| `--test`  | filename                 | Path to test txt file                                             |
| `--rul`   | filename                 | Path to RUL txt file                                              |
| `--model` | `ml`, `dl`, `xgb`, `all` | `ml`=Random Forest, `dl`=LSTM, `xgb`=XGBoost, `all`=run all three |

## Results

All datasets were evaluated on NASA CMAPSS FD001–FD004. Failure threshold: RUL ≤ 30 cycles.

### Baseline benchmark (single train/test split)

The original benchmark uses a fixed train/test split with default hyperparameters. Under this protocol, LSTM leads on FD001–FD003.

#### Recall

| Dataset | RF   | XGBoost  | LSTM     |
| ------- | ---- | -------- | -------- |
| FD001   | 0.68 | 0.72     | **0.80** |
| FD002   | 0.90 | 0.93     | **0.97** |
| FD003   | 0.75 | 0.65     | **0.95** |
| FD004   | 0.74 | **0.75** | 0.66     |

#### Precision

| Dataset | RF        | XGBoost   | LSTM      |
| ------- | --------- | --------- | --------- |
| FD001   | 0.944     | **0.947** | 0.909     |
| FD002   | 0.965     | 0.934     | **0.967** |
| FD003   | 0.938     | 0.929     | **0.950** |
| FD004   | **0.929** | 0.909     | 0.854     |

#### ROC-AUC

| Dataset | RF        | XGBoost   | LSTM      |
| ------- | --------- | --------- | --------- |
| FD001   | 0.974     | 0.970     | **0.990** |
| FD002   | 0.990     | 0.988     | **0.998** |
| FD003   | 0.989     | 0.985     | **0.995** |
| FD004   | **0.979** | **0.979** | 0.977     |

### Tuned nested-CV results (5-fold Group CV, 50 Optuna trials, 188 features)

After tuning Random Forest (RF) and XGBoost models with Optuna in a nested cross-validation (CV) loop on the engineered 188-feature set, and grouping CV by engine to prevent data leakage, **classical machine learning methods match or surpass the fixed-hyperparameter LSTM model in recall**. They also demonstrate greater stability.

#### FD001 — 100 engines, single operating condition

| Model   | Precision       | Recall              | ROC-AUC             |
| ------- | --------------- | ------------------- | ------------------- |
| RF      | 0.9421 ± 0.0104 | 0.8890 ± 0.0101     | **0.9962 ± 0.0007** |
| XGBoost | 0.9406 ± 0.0108 | **0.8949 ± 0.0149** | 0.9958 ± 0.0007     |
| LSTM    | 0.9337 ± 0.0222 | 0.8677 ± 0.0220     | 0.9949 ± 0.0007     |

#### FD002 — 260 engines, 6 operating conditions

| Model   | Precision           | Recall              | ROC-AUC             |
| ------- | ------------------- | ------------------- | ------------------- |
| RF      | **0.9414 ± 0.0045** | 0.8843 ± 0.0073     | **0.9964 ± 0.0001** |
| XGBoost | 0.9351 ± 0.0082     | **0.8855 ± 0.0110** | 0.9962 ± 0.0005     |
| LSTM    | 0.8748 ± 0.0617     | 0.7586 ± 0.1292     | 0.9641 ± 0.0113     |

Key observations:

- On FD001, RF / XGBoost / LSTM are within ~0.03 recall of each other; tree models edge out LSTM by **+0.021 (RF)** and **+0.027 (XGBoost)**.
- On FD002 the gap **widens to +0.126 (RF) and +0.127 (XGBoost)** over LSTM. LSTM fold 5 recall collapsed to 0.519 (std 0.129), confirming the model struggles with mixed operating conditions at fixed hyperparameters.
- RF and XGBoost are extremely stable across folds (std < 0.011 on every metric on both datasets).
- FD003 / FD004 tuned runs are pending; see `experiments/tuned_results_FD001_FD002.txt` for full per-fold detail.

### Statistical significance (RF vs XGBoost, all four datasets)

5-fold Group CV, α = 0.05. LSTM was skipped in this run (TensorFlow unavailable on the host). Source: `experiments/statistical_report.txt`.

| Dataset | Precision (RF / XGB) | Paired t-test p (corr.) | McNemar p    |
| ------- | -------------------- | ----------------------- | ------------ |
| FD001   | 0.9215 / 0.9002      | **0.0372\***            | 0.7135       |
| FD002   | 0.9085 / 0.8883      | **0.0002\***            | 0.1333       |
| FD003   | 0.9150 / 0.9023      | **0.0314\***            | 0.6136       |
| FD004   | 0.9065 / 0.8895      | **0.0001\***            | **0.0015\*** |

\* = p < 0.05.

- **RF beats XGBoost on precision on every dataset**, and the paired t-test is significant on all four (Bonferroni-corrected). Wilcoxon p = 0.0625 on each (the floor for n = 5), so it cannot reject at α = 0.05.
- ROC-AUC differences between RF and XGBoost are **not significant on any dataset** — the two models separate failures from non-failures equally well.
- McNemar on pooled per-sample predictions only flags FD004 as significant, confirming that FD004's mixed conditions surface real per-sample disagreements between the two models.

### Summary

- Under a **fixed-split / default-hyperparameter** protocol, LSTM looks dominant on FD001–FD003; FD004 (6 operating conditions + 2 fault modes) is the hardest case and is owned by RF / XGBoost.
- Under a **nested-CV + 188 engineered features + Optuna tuning** protocol, RF and XGBoost match or beat LSTM on recall and are dramatically more stable. LSTM at fixed hyperparameters underperforms badly on FD002 multi-condition data.
- **RF beats XGBoost on precision with statistical significance on all four datasets**; ROC-AUC is statistically indistinguishable between the two.
- For predictive maintenance, where missing a failure is costlier than a false alarm, recall-optimized RF or XGBoost is the recommended default; LSTM is only competitive once given more epochs, architecture tuning, or operating-condition-aware normalization.
- Per-run regression metrics (RMSE, MAE, R², NASA score) are tracked in `experiments/logs.csv`; consolidated tables live in `experiments/results_summary.md`; full statistical detail in `experiments/statistical_report.txt` and `experiments/tuned_results_FD001_FD002.txt`.

## Testing

```bash
pytest tests/ -v
```

## Configuration

Edit `configs/config.yaml` to adjust model hyperparameters, data splits, feature columns, and file paths.

## Author

José Eduardo Herrera. Feedback welcome via GitHub issues.
