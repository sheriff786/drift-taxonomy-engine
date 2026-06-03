"""Prediction Explorer Page — Enhanced with model selection & visualization."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from pathlib import Path

st.set_page_config(page_title="Prediction Explorer", layout="wide")

st.markdown("""
<style>
    .fraud-alert { background: linear-gradient(135deg, #7f1d1d, #991b1b); border: 1px solid #ef4444;
                   border-radius: 12px; padding: 20px; text-align: center; }
    .legit-alert { background: linear-gradient(135deg, #064e3b, #065f46); border: 1px solid #22c55e;
                   border-radius: 12px; padding: 20px; text-align: center; }
</style>
""", unsafe_allow_html=True)

API_BASE = "http://localhost:8000/api/v1"
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"

st.title("🔎 Prediction Explorer")
st.caption("Score transactions with model selection · Compare model predictions · Batch analysis")
st.markdown("---")

# Feature name mapping for display
FEATURE_NAMES = [
    "card_auth_maturity", "velocity_anomaly", "spending_pattern_match",
    "high_risk_merchant_flag", "channel_risk_score", "geo_distance_indicator",
    "location_consistency", "time_deviation", "txn_frequency_normality",
    "behavioral_consistency", "unusual_activity_flag", "address_verification_score",
    "account_balance_ratio", "cardholder_verification", "payment_method_risk",
    "merchant_reputation", "transaction_legitimacy", "auth_strength_score",
    "ip_risk_score", "device_fingerprint", "cross_border_indicator",
    "recurring_pattern", "entry_mode_risk", "billing_match_score",
    "card_present_indicator", "refund_history", "pin_verification_result",
    "decline_history_score", "transaction_amount", "transaction_time",
]

# --- Model Selection ---
available_models = ["random_forest", "xgboost", "lightgbm", "logistic_regression"]
try:
    resp = requests.get(f"{API_BASE}/models/available", timeout=5)
    if resp.status_code == 200:
        available_models = resp.json().get("available_models", available_models)
except Exception:
    pass

selected_model = st.sidebar.selectbox("🤖 Model", ["default (best)"] + available_models)
model_param = None if selected_model.startswith("default") else selected_model

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Actions")
use_random = st.sidebar.button("🎲 Generate Random Transaction")

# --- Single Transaction ---
st.subheader("🧪 Single Transaction Scoring")

col_form, col_result = st.columns([2, 1])

with col_form:
    with st.form("predict_form"):
        st.markdown("**Transaction Features (Domain Names)**")

        # Pre-populate with random data if requested
        default_vals = {}
        if use_random:
            np.random.seed(None)
            for feat in FEATURE_NAMES:
                default_vals[feat] = float(np.random.randn())

        features = {}
        cols = st.columns(4)
        for i, feat in enumerate(FEATURE_NAMES):
            col_idx = i % 4
            with cols[col_idx]:
                features[feat] = st.number_input(
                    feat.replace("_", " ").title()[:20],
                    value=default_vals.get(feat, 0.0),
                    key=f"feat_{i}", format="%.4f",
                    help=feat,
                )

        submitted = st.form_submit_button("🚀 Score Transaction", type="primary")

with col_result:
    st.markdown("### Result")

    if submitted:
        payload = {"samples": [features]}
        if model_param:
            payload["model_name"] = model_param

        try:
            resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                prob = result["probabilities"][0]
                pred = result["predictions"][0]
                model_used = result.get("model_name", "default")

                if pred == 1:
                    st.markdown(f"""
                    <div class="fraud-alert">
                        <div style="font-size:36px">🚨</div>
                        <div style="font-size:20px;font-weight:600;color:#fecaca;margin-top:8px">FRAUD DETECTED</div>
                        <div style="font-size:28px;font-weight:700;color:#ef4444;margin-top:6px">{prob:.4f}</div>
                        <div style="font-size:11px;color:#fca5a5;margin-top:4px">Model: {model_used}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="legit-alert">
                        <div style="font-size:36px">✅</div>
                        <div style="font-size:20px;font-weight:600;color:#d1fae5;margin-top:8px">LEGITIMATE</div>
                        <div style="font-size:28px;font-weight:700;color:#22c55e;margin-top:6px">{prob:.4f}</div>
                        <div style="font-size:11px;color:#86efac;margin-top:4px">Model: {model_used}</div>
                    </div>""", unsafe_allow_html=True)

                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob,
                    title={"text": "Fraud Probability"},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "#ef4444" if prob > 0.5 else "#22c55e"},
                        "steps": [
                            {"range": [0, 0.3], "color": "#d1fae5"},
                            {"range": [0.3, 0.7], "color": "#fef3c7"},
                            {"range": [0.7, 1], "color": "#fecaca"},
                        ],
                        "threshold": {"line": {"color": "#ef4444", "width": 2}, "value": 0.5},
                    },
                ))
                fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.error(f"Prediction failed: {resp.text}")
        except requests.ConnectionError:
            st.error("API not running.")
    else:
        st.info("Submit a transaction to see prediction results.")

# --- Multi-Model Comparison ---
st.markdown("---")
st.subheader("🔀 Multi-Model Comparison")
st.caption("Score the same transaction across all models to compare predictions")

if st.button("⚡ Compare All Models") and submitted:
    comparison = []
    for model in available_models:
        try:
            payload = {"samples": [features], "model_name": model}
            resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
            if resp.status_code == 200:
                r = resp.json()
                comparison.append({
                    "Model": model,
                    "Prediction": "🔴 Fraud" if r["predictions"][0] == 1 else "🟢 Legit",
                    "Probability": round(r["probabilities"][0], 6),
                })
        except Exception:
            comparison.append({"Model": model, "Prediction": "❌ Error", "Probability": 0})

    if comparison:
        cmp_df = pd.DataFrame(comparison).sort_values("Probability", ascending=False)
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        fig_cmp = px.bar(cmp_df, x="Model", y="Probability", color="Probability",
                         color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"], range_color=[0, 1])
        fig_cmp.add_hline(y=0.5, line_dash="dash", line_color="#ef4444")
        fig_cmp.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_cmp, use_container_width=True)

# --- Batch Scoring ---
st.markdown("---")
st.subheader("📁 Batch Scoring")

batch_file = st.file_uploader("Upload transactions CSV", type=["csv"])
if batch_file is not None:
    batch_df = pd.read_csv(batch_file)
    st.write(f"Loaded **{len(batch_df):,}** transactions")

    if st.button("🚀 Score Batch", type="primary"):
        with st.spinner(f"Scoring {len(batch_df):,} transactions with {model_param or 'best model'}..."):
            samples = batch_df.to_dict(orient="records")
            payload = {"samples": samples}
            if model_param:
                payload["model_name"] = model_param

            try:
                resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=60)
                if resp.status_code == 200:
                    result = resp.json()
                    batch_df["fraud_probability"] = result["probabilities"]
                    batch_df["prediction"] = result["predictions"]
                    batch_df["label"] = batch_df["prediction"].map({0: "Legit", 1: "Fraud"})

                    # Stats
                    n_fraud = sum(result["predictions"])
                    n_total = len(result["predictions"])

                    bc1, bc2, bc3 = st.columns(3)
                    bc1.metric("Total Scored", f"{n_total:,}")
                    bc2.metric("Fraud Detected", f"{n_fraud:,}")
                    bc3.metric("Fraud Rate", f"{n_fraud/n_total*100:.2f}%")

                    # Distribution
                    fig_dist = px.histogram(batch_df, x="fraud_probability", color="label",
                                            nbins=50, color_discrete_map={"Fraud": "#ef4444", "Legit": "#22c55e"})
                    fig_dist.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_dist, use_container_width=True)

                    st.dataframe(batch_df[["fraud_probability", "prediction"]].describe())
                else:
                    st.error(f"Batch scoring failed: {resp.text}")
            except requests.ConnectionError:
                st.error("API not running.")
