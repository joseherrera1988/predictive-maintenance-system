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
| Model         | ROC-AUC | Precision | Recall |
|---------------|--------|-----------|--------|
| Random Forest | 0.XX   | 0.XX      | 0.XX   |
| XGBoost       | 0.XX   | 0.XX      | 0.XX   |
| LSTM          | 0.XX   | 0.XX      | 0.XX   |

## Dashboard
![Dashboard](assets/dashboard.png)

## How to Run
```bash
pip install -r requirements.txt
python main.py
streamlit run dashboards/streamlit_app.py
