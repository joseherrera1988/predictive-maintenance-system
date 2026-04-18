# predictive-maintenance-system
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

## Results
| Model          | ROC-AUC | Recall |
|----------------|--------|--------|
| Random Forest  | 0.87   | 0.81   |
| XGBoost        | 0.89   | 0.84   |
| LSTM           | 0.91   | 0.86   |

## Dashboard
![Dashboard](assets/dashboard.png)

## How to Run
```bash
pip install -r requirements.txt
python main.py
streamlit run dashboards/streamlit_app.py
