"""Model Performance Monitoring Page — Enhanced with MLflow and comparison charts."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from pathlib import Path

st.set_page_config(page_title="Model Performance", layout="wide")

# CSS
st.markdown("""
<style>
    .model-card { background: #1e1e2e; border: 1px solid #3d3d5c; border-radius: 10px; padding: 16px; margin-bottom: 12px; }
    .model-name { font-size: 16px; font-weight: 600; color: #e2e8f0; }
    .model-meta { font-size: 11px; color: #9ca3af; }
    .champion-badge { background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

API_BASE = "http://localhost:8000/api/v1"
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"


def load_model_registry():
    reg_path = ARTIFACTS_DIR / "models" / "registry.json"
    if reg_path.exists():
        with open(reg_path) as f:
            return json.load(f)
    return None


st.title("🏆 Model Performance & Comparison")
st.caption("Track model metrics · Compare candidates · Monitor production model health")
st.markdown("---")

registry = load_model_registry()

if registry and registry.get("models"):
    models = registry["models"]
    latest = registry.get("latest", "")

    # --- Model Comparison Table ---
    st.subheader("📊 Model Comparison Matrix")

    comparison_data = []
    for name, info in models.items():
        metrics = info.get("metrics", {})
        comparison_data.append({
            "Model": name,
            "AUPRC": round(metrics.get("auprc", 0), 4),
            "AUROC": round(metrics.get("auroc", 0), 4),
            "F1": round(metrics.get("f1", 0), 4),
            "Precision": round(metrics.get("precision", 0), 4),
            "Recall": round(metrics.get("recall", 0), 4),
            "Status": "🏆 Production" if name == latest else "📦 Candidate",
            "Version": info.get("version", "?"),
        })

    comp_df = pd.DataFrame(comparison_data).sort_values("AUPRC", ascending=False)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # --- Grouped Bar Chart ---
    st.markdown("")
    chart_col, radar_col = st.columns(2)

    with chart_col:
        st.markdown("#### Metric Comparison")
        metrics_to_plot = ["AUPRC", "AUROC", "F1", "Precision", "Recall"]
        fig_bar = px.bar(
            comp_df.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metric", value_name="Score"),
            x="Model", y="Score", color="Metric", barmode="group",
            color_discrete_sequence=["#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6", "#ef4444"],
        )
        fig_bar.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_bar, use_container_width=True)

    with radar_col:
        st.markdown("#### Performance Radar")
        fig_radar = go.Figure()
        colors = ["#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6"]
        for i, row in comp_df.iterrows():
            values = [row["AUPRC"], row["AUROC"], row["F1"], row["Precision"], row["Recall"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=metrics_to_plot + [metrics_to_plot[0]],
                name=row["Model"],
                line=dict(color=colors[i % len(colors)]),
                fill="toself" if row["Model"] == latest else None,
                opacity=1 if row["Model"] == latest else 0.5,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=350, margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True, legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- Individual Model Cards ---
    st.markdown("---")
    st.subheader("📋 Model Details")

    for name, info in models.items():
        metrics = info.get("metrics", {})
        is_champion = name == latest

        with st.expander(f"{'🏆' if is_champion else '📦'} {name} {'(Production)' if is_champion else ''}", expanded=is_champion):
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("AUPRC", f"{metrics.get('auprc', 0):.4f}")
            m2.metric("AUROC", f"{metrics.get('auroc', 0):.4f}")
            m3.metric("F1", f"{metrics.get('f1', 0):.4f}")
            m4.metric("Precision", f"{metrics.get('precision', 0):.4f}")
            m5.metric("Recall", f"{metrics.get('recall', 0):.4f}")

            st.markdown(f"**Version:** `{info.get('version', 'N/A')}` | **Registered:** `{info.get('registered_at', 'N/A')}`")

            if info.get("hyperparameters"):
                st.markdown("**Hyperparameters:**")
                st.json(info["hyperparameters"])

    # --- MLflow Link ---
    st.markdown("---")
    st.info("🧪 **MLflow Tracking Server**: Open [http://localhost:5000](http://localhost:5000) for full experiment tracking, artifact storage, and model versioning.")

else:
    st.warning("No models registered yet.")
    st.info("Run the training pipeline first: `python -m pipelines.training_pipeline`")
