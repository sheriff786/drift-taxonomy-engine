"""Prediction Explorer Page."""

import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Prediction Explorer", layout="wide")
st.title("Prediction Explorer")

API_BASE = "http://localhost:8000/api/v1"

st.markdown("## Single Transaction Scoring")

with st.form("predict_form"):
    st.markdown("Enter transaction features (V1-V28 + scaled Amount/Time):")

    cols = st.columns(5)
    features = {}
    for i in range(1, 29):
        col_idx = (i - 1) % 5
        with cols[col_idx]:
            features[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v{i}")

    col_a, col_t = st.columns(2)
    with col_a:
        features["Amount_scaled"] = st.number_input("Amount (scaled)", value=0.0)
    with col_t:
        features["Time_scaled"] = st.number_input("Time (scaled)", value=0.0)

    submitted = st.form_submit_button("Score Transaction")

if submitted:
    try:
        resp = requests.post(
            f"{API_BASE}/predict",
            json={"samples": [features]},
            timeout=10,
        )
        if resp.status_code == 200:
            result = resp.json()
            prob = result["probabilities"][0]
            pred = result["predictions"][0]

            if pred == 1:
                st.error(f"**FRAUD DETECTED** - Probability: {prob:.4f}")
            else:
                st.success(f"**Legitimate** - Fraud Probability: {prob:.4f}")
        else:
            st.error(f"Prediction failed: {resp.text}")
    except requests.ConnectionError:
        st.error("API not running. Start the API server first.")

st.markdown("---")
st.markdown("## Batch Scoring")
st.info("Upload a CSV file to score multiple transactions.")

batch_file = st.file_uploader("Upload transactions (CSV)", type=["csv"])
if batch_file is not None:
    batch_df = pd.read_csv(batch_file)
    st.write(f"Loaded {len(batch_df)} transactions.")

    if st.button("Score Batch"):
        samples = batch_df.to_dict(orient="records")
        try:
            resp = requests.post(
                f"{API_BASE}/predict",
                json={"samples": samples},
                timeout=30,
            )
            if resp.status_code == 200:
                result = resp.json()
                batch_df["fraud_probability"] = result["probabilities"]
                batch_df["prediction"] = result["predictions"]
                st.dataframe(batch_df[["fraud_probability", "prediction"]].describe())
                st.dataframe(batch_df)
        except requests.ConnectionError:
            st.error("API not running.")
