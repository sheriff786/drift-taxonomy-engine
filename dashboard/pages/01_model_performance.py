"""Model Performance Monitoring Page."""

import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("Model Performance Monitoring")

API_BASE = "http://localhost:8000/api/v1"

st.markdown("## Registered Models")

try:
    resp = requests.get(f"{API_BASE}/models", timeout=5)
    if resp.status_code == 200:
        data = resp.json()
        st.write(f"**Total models registered:** {data['total']}")
        for model_name in data["models"]:
            with st.expander(f"Model: {model_name}"):
                info_resp = requests.get(f"{API_BASE}/models/{model_name}", timeout=5)
                if info_resp.status_code == 200:
                    info = info_resp.json()
                    st.json(info)
    else:
        st.warning("Could not fetch models from API.")
except requests.ConnectionError:
    st.error("API not running. Start the API server first: `uvicorn api.main:app`")

st.markdown("---")
st.markdown("## Performance Metrics")
st.info("Connect to MLflow tracking server for historical metrics visualization.")
