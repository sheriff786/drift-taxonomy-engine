"""Drift Monitoring Dashboard Page."""

import streamlit as st
import pandas as pd
import requests
import json

st.set_page_config(page_title="Drift Monitor", layout="wide")
st.title("Drift Monitoring Dashboard")

API_BASE = "http://localhost:8000/api/v1"

st.markdown("## Current Drift Status")

try:
    resp = requests.get(f"{API_BASE}/drift/status", timeout=5)
    if resp.status_code == 200:
        status = resp.json()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Drift Type", status["current_drift_type"])
        with col2:
            st.metric("Severity", status["current_severity"])
        with col3:
            st.metric("Checks (24h)", status["checks_run_24h"])
    else:
        st.warning("Could not fetch drift status.")
except requests.ConnectionError:
    st.error("API not running. Start the API server first.")

st.markdown("---")
st.markdown("## Run Manual Drift Check")
st.info("Upload a CSV of current production data to diagnose drift against the reference baseline.")

uploaded_file = st.file_uploader("Upload current data (CSV)", type=["csv"])

if uploaded_file is not None:
    current_df = pd.read_csv(uploaded_file)
    st.write(f"Loaded {len(current_df)} samples with {len(current_df.columns)} features.")

    if st.button("Run Drift Diagnosis"):
        with st.spinner("Running drift taxonomy engine..."):
            samples = current_df.to_dict(orient="records")
            try:
                resp = requests.post(
                    f"{API_BASE}/drift/diagnose",
                    json={"current_samples": samples},
                    timeout=30,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.success("Drift diagnosis complete!")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Type", result["drift_type"])
                    with col2:
                        st.metric("Severity", result["severity"])
                    with col3:
                        st.metric("Action", result["action"])
                    with col4:
                        st.metric("Urgency", f"{result['urgency_hours']}h")

                    st.markdown("### Scores")
                    scores_df = pd.DataFrame([{
                        "Covariate": result["covariate_score"],
                        "Concept": result["concept_score"],
                        "Pipeline": result["pipeline_score"],
                        "Target": result["target_score"],
                    }])
                    st.bar_chart(scores_df.T)

                    if result["drifted_features"]:
                        st.markdown("### Drifted Features")
                        st.write(result["drifted_features"])

                    if result["playbook"]:
                        st.markdown("### Response Playbook")
                        st.json(result["playbook"])
                else:
                    st.error(f"Diagnosis failed: {resp.text}")
            except requests.ConnectionError:
                st.error("API not running.")
