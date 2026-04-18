import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")

st.title("Predictive Maintenance AI Dashboard")

# --- Experiment logs ---
st.header("Experiment Logs")
logs_path = Path("experiments/logs.csv")
if logs_path.exists() and logs_path.stat().st_size > 0:
    df = pd.read_csv(logs_path)
    st.dataframe(df, use_container_width=True)
else:
    st.info("No experiments logged yet.")

# --- Model metadata ---
st.header("Saved Models")
meta_path = Path("models/metadata.json")
if meta_path.exists():
    with open(meta_path) as f:
        metadata = json.load(f)
    if metadata["models"]:
        st.table(pd.DataFrame(metadata["models"]))
    else:
        st.info("No models saved yet.")
else:
    st.info("metadata.json not found.")

# --- Run prediction ---
st.header("Run Prediction")
uploaded = st.file_uploader("Upload CSV for prediction", type=["csv"])
if uploaded:
    df_input = pd.read_csv(uploaded)
    st.write("Preview:", df_input.head())
    st.warning("Connect a trained model to generate predictions.")
