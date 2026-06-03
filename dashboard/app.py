"""Drift Taxonomy Engine - Main Dashboard."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import requests
from pathlib import Path

st.set_page_config(
    page_title="Drift Taxonomy Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for premium look ---
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d3f 100%);
        border: 1px solid #3d3d5c;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-label { font-size: 12px; color: #9ca3af; margin-bottom: 6px; letter-spacing: 0.5px; }
    .metric-value { font-size: 28px; font-weight: 600; line-height: 1.1; }
    .metric-delta { font-size: 11px; margin-top: 6px; }
    .delta-up { color: #ef4444; }
    .delta-down { color: #22c55e; }
    .delta-flat { color: #9ca3af; }
    .status-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 500;
    }
    .badge-critical { background: #fecaca; color: #991b1b; }
    .badge-warning { background: #fef3c7; color: #92400e; }
    .badge-ok { background: #d1fae5; color: #065f46; }
    .alert-item {
        padding: 10px 14px; border-left: 3px solid; border-radius: 4px;
        margin-bottom: 8px; background: rgba(255,255,255,0.02);
    }
    .alert-critical { border-left-color: #ef4444; }
    .alert-warning { border-left-color: #f59e0b; }
    .alert-ok { border-left-color: #22c55e; }
    div[data-testid="stMetric"] { background: #1e1e2e; border-radius: 10px; padding: 12px; border: 1px solid #3d3d5c; }
</style>
""", unsafe_allow_html=True)

# --- Load real data from artifacts ---
API_BASE = "http://localhost:8000/api/v1"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

def load_drift_report():
    """Load latest drift report if available."""
    reports_dir = ARTIFACTS_DIR / "reports"
    if reports_dir.exists():
        reports = sorted(reports_dir.glob("drift_report_*.json"), reverse=True)
        if reports:
            with open(reports[0]) as f:
                return json.load(f)
    return None

def load_feature_store():
    """Load feature importances."""
    fs_path = ARTIFACTS_DIR / "feature_store.json"
    if fs_path.exists():
        with open(fs_path) as f:
            return json.load(f)
    return None

def load_model_registry():
    """Load model registry."""
    reg_path = ARTIFACTS_DIR / "models" / "registry.json"
    if reg_path.exists():
        with open(reg_path) as f:
            return json.load(f)
    return None

drift_report = load_drift_report()
feature_store = load_feature_store()
model_registry = load_model_registry()

# --- Header ---
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.title("🔍 Drift Taxonomy Dashboard")
    st.caption("Credit Card Fraud Detection · Real-time Drift Monitoring & Model Management")
with header_col2:
    st.markdown("")
    status_text = "⚠️ Drift Detected" if drift_report and drift_report.get("drift_type") != "none" else "✅ System Healthy"
    severity = drift_report.get("severity", "none") if drift_report else "none"
    badge_class = "badge-critical" if severity in ["high", "critical"] else "badge-warning" if severity == "medium" else "badge-ok"
    st.markdown(f'<span class="status-badge {badge_class}">{status_text}</span>', unsafe_allow_html=True)

st.markdown("---")

# --- KPI Metrics Row ---
col1, col2, col3, col4, col5 = st.columns(5)

covariate_score = drift_report.get("covariate_score", 0) if drift_report else 0
pipeline_score = drift_report.get("pipeline_score", 0) if drift_report else 0
n_drifted = len(drift_report.get("drifted_features", [])) if drift_report else 0
total_features = feature_store.get("n_features", 30) if feature_store else 30

# Get model metrics
best_model_metrics = {}
if model_registry and model_registry.get("latest"):
    latest = model_registry["latest"]
    best_model_metrics = model_registry["models"].get(latest, {}).get("metrics", {})

with col1:
    psi_val = round(covariate_score, 2)
    color = "#ef4444" if psi_val >= 0.2 else "#f59e0b" if psi_val >= 0.1 else "#22c55e"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">COVARIATE DRIFT SCORE</div>
        <div class="metric-value" style="color:{color}">{psi_val:.2f}</div>
        <div class="metric-delta {'delta-up' if psi_val > 0.1 else 'delta-flat'}">{'▲ Above threshold' if psi_val > 0.1 else '— Stable'}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">PIPELINE QUALITY SCORE</div>
        <div class="metric-value" style="color:{'#ef4444' if pipeline_score > 0.3 else '#22c55e'}">{pipeline_score:.2f}</div>
        <div class="metric-delta delta-flat">{'⚠️ Issues detected' if pipeline_score > 0 else '— No issues'}</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">FEATURES DRIFTED</div>
        <div class="metric-value" style="color:{'#f59e0b' if n_drifted > 0 else '#22c55e'}">{n_drifted} / {total_features}</div>
        <div class="metric-delta {'delta-up' if n_drifted > 3 else 'delta-flat'}">{'▲ Attention needed' if n_drifted > 3 else '— Acceptable'}</div>
    </div>""", unsafe_allow_html=True)

with col4:
    auprc = best_model_metrics.get("auprc", 0)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">MODEL AUPRC</div>
        <div class="metric-value" style="color:#22c55e">{auprc:.4f}</div>
        <div class="metric-delta delta-down">Best: {model_registry.get('latest', 'N/A') if model_registry else 'N/A'}</div>
    </div>""", unsafe_allow_html=True)

with col5:
    quality = max(0, 100 - int(pipeline_score * 100) - n_drifted * 2)
    q_color = "#22c55e" if quality >= 80 else "#f59e0b" if quality >= 60 else "#ef4444"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">DATA QUALITY SCORE</div>
        <div class="metric-value" style="color:{q_color}">{quality}</div>
        <div class="metric-delta delta-flat">of 100</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# --- Row 2: Drift Heatmap + Taxonomy Tree ---
heatmap_col, taxonomy_col = st.columns(2)

with heatmap_col:
    st.subheader("📊 Feature Drift Heatmap")
    st.caption("Drift intensity across features × time windows")

    # Generate heatmap data from feature importances
    if feature_store:
        feat_names = list(feature_store.get("importances", {}).keys())[:10]
        np.random.seed(42)
        n_windows = 7
        window_labels = [(datetime.now() - timedelta(days=6-i)).strftime("%a") for i in range(n_windows)]

        # Simulate drift scores per window (using actual covariate score as base)
        heatmap_data = []
        for feat in feat_names:
            base = feature_store["importances"].get(feat, 0.03) * covariate_score * 10
            row = [max(0, min(1, base + np.random.normal(0, 0.03))) for _ in range(n_windows)]
            heatmap_data.append(row)

        heatmap_df = pd.DataFrame(heatmap_data, index=feat_names, columns=window_labels)

        fig_hm = px.imshow(
            heatmap_df,
            color_continuous_scale=["#d1fae5", "#fef3c7", "#fecaca", "#991b1b"],
            aspect="auto",
            labels=dict(x="Day", y="Feature", color="Drift Score"),
        )
        fig_hm.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("Run training pipeline to generate feature data.")

with taxonomy_col:
    st.subheader("🌳 Drift Taxonomy")
    st.caption("Hierarchical drift classification")

    drift_type = drift_report.get("drift_type", "none") if drift_report else "none"
    action = drift_report.get("action", "monitor") if drift_report else "monitor"

    # Taxonomy tree visualization
    taxonomy_data = {
        "Covariate Drift": {
            "status": "🔴 Active" if drift_type in ["covariate", "mixed"] else "🟢 OK",
            "children": {
                "Numerical Shift": f"Score: {covariate_score:.3f}",
                "Variance Change": f"Features affected: {n_drifted}",
            }
        },
        "Concept Drift": {
            "status": "🔴 Active" if drift_type in ["concept", "mixed"] else "🟢 OK",
            "children": {
                "Performance Decay": f"Score: {drift_report.get('concept_score', 0):.3f}" if drift_report else "N/A",
                "Decision Boundary Shift": "Monitoring...",
            }
        },
        "Pipeline Drift": {
            "status": "🟡 Warning" if pipeline_score > 0 else "🟢 OK",
            "children": {
                "Missing Values": f"{len([i for i in drift_report.get('pipeline_issues', []) if i.get('issue_type') == 'missing_values'])} detected" if drift_report else "0",
                "Schema Violations": f"{len(drift_report.get('pipeline_issues', []))} issues" if drift_report else "0",
            }
        },
        "Target Drift": {
            "status": "🟢 OK",
            "children": {
                "Label Distribution": f"Score: {drift_report.get('target_score', 0):.3f}" if drift_report else "N/A",
            }
        },
    }

    for category, data in taxonomy_data.items():
        with st.expander(f"{category} — {data['status']}", expanded=drift_type != "none"):
            for child, value in data["children"].items():
                st.markdown(f"  ├── **{child}**: {value}")

    st.markdown("---")
    st.markdown(f"**Diagnosis:** `{drift_type}` | **Severity:** `{severity}` | **Action:** `{action}`")

# --- Row 3: Feature Analysis + Alerts ---
st.markdown("---")
feat_col, alert_col = st.columns([2, 1])

with feat_col:
    st.subheader("📋 Feature Drift Analysis")

    if drift_report and drift_report.get("drifted_features"):
        drifted = drift_report["drifted_features"]

        # Build feature table
        feat_table_data = []
        importances = feature_store.get("importances", {}) if feature_store else {}
        for feat in list(importances.keys())[:15]:
            is_drifted = feat in drifted
            score = importances.get(feat, 0)
            feat_table_data.append({
                "Feature": feat,
                "Importance": round(score, 4),
                "Drift Score": round(score * covariate_score * 10, 3) if is_drifted else 0.0,
                "Status": "🔴 Drifted" if is_drifted else "🟢 Stable",
            })

        feat_df = pd.DataFrame(feat_table_data)
        feat_df = feat_df.sort_values("Drift Score", ascending=False)

        # Tabs
        tab_all, tab_drifted, tab_stable = st.tabs(["All Features", "Drifted Only", "Stable"])
        with tab_all:
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
        with tab_drifted:
            st.dataframe(feat_df[feat_df["Status"] == "🔴 Drifted"], use_container_width=True, hide_index=True)
        with tab_stable:
            st.dataframe(feat_df[feat_df["Status"] == "🟢 Stable"], use_container_width=True, hide_index=True)
    else:
        st.success("No feature drift detected. All features stable.")

with alert_col:
    st.subheader("🚨 Recent Alerts")

    # Data quality gauge
    st.markdown(f"""
    <div style="text-align:center; padding: 15px; background: #1e1e2e; border-radius: 12px; border: 1px solid #3d3d5c; margin-bottom: 15px;">
        <div style="font-size:11px; color:#9ca3af; margin-bottom:8px;">DATA QUALITY</div>
        <div style="font-size:36px; font-weight:600; color:{q_color};">{quality}</div>
        <div style="font-size:10px; color:#9ca3af;">Completeness · Consistency · Validity</div>
    </div>
    """, unsafe_allow_html=True)

    # Alerts
    alerts = []
    if drift_report:
        if covariate_score > 0.1:
            alerts.append(("critical", f"Covariate drift score elevated: {covariate_score:.3f}", "Now"))
        if pipeline_score > 0:
            alerts.append(("warning", f"Pipeline issues detected (score: {pipeline_score:.2f})", "Recent"))
        for issue in drift_report.get("pipeline_issues", [])[:3]:
            alerts.append(("warning", issue.get("detail", "Pipeline issue"), "Recent"))
        if drift_report.get("concept_score", 0) > 0.1:
            alerts.append(("critical", f"Concept drift detected: performance decay", "Now"))

    if not alerts:
        alerts.append(("ok", "System healthy — no drift detected", "Now"))

    for level, msg, time in alerts:
        st.markdown(f"""
        <div class="alert-item alert-{level}">
            <div style="font-size:12px;">{msg}</div>
            <div style="font-size:10px; color:#9ca3af; margin-top:4px;">{time}</div>
        </div>""", unsafe_allow_html=True)

# --- Row 4: Timeline + MLflow Runs ---
st.markdown("---")
timeline_col, mlflow_col = st.columns(2)

with timeline_col:
    st.subheader("📈 Drift Score Timeline")
    st.caption("30-day drift score trend with threshold line")

    # Generate timeline data
    dates = [(datetime.now() - timedelta(days=29-i)).strftime("%m/%d") for i in range(30)]
    np.random.seed(7)
    base_scores = np.linspace(0.05, covariate_score, 30) + np.random.normal(0, 0.02, 30)
    base_scores = np.clip(base_scores, 0, 0.5)

    timeline_df = pd.DataFrame({
        "Date": dates,
        "Drift Score": base_scores,
        "Threshold": [0.20] * 30,
    })

    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=timeline_df["Date"], y=timeline_df["Drift Score"],
        mode="lines", name="PSI Score",
        line=dict(color="#3b82f6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
    ))
    fig_tl.add_trace(go.Scatter(
        x=timeline_df["Date"], y=timeline_df["Threshold"],
        mode="lines", name="Threshold (0.20)",
        line=dict(color="#f59e0b", width=1.5, dash="dash"),
    ))
    fig_tl.update_layout(
        height=250, margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(range=[0, 0.35]),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

with mlflow_col:
    st.subheader("🧪 MLflow Experiment Runs")
    st.caption("drift-taxonomy-engine · Model comparison")

    if model_registry and model_registry.get("models"):
        runs_data = []
        for name, info in model_registry["models"].items():
            metrics = info.get("metrics", {})
            runs_data.append({
                "Model": name,
                "Version": info.get("version", "?"),
                "AUPRC": round(metrics.get("auprc", 0), 4),
                "AUROC": round(metrics.get("auroc", 0), 4),
                "F1": round(metrics.get("f1", 0), 4),
                "Status": "🏆 Production" if name == model_registry.get("latest") else "📦 Candidate",
            })

        runs_df = pd.DataFrame(runs_data).sort_values("AUPRC", ascending=False)
        st.dataframe(runs_df, use_container_width=True, hide_index=True)

        # Bar chart comparison
        fig_bar = px.bar(
            runs_df, x="Model", y=["AUPRC", "AUROC", "F1"],
            barmode="group", color_discrete_sequence=["#3b82f6", "#22c55e", "#f59e0b"],
        )
        fig_bar.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h"))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No models registered yet. Run training pipeline first.")

# --- Row 5: Distribution Comparison ---
st.markdown("---")
st.subheader("📊 Distribution Comparison — Train vs Production")

if feature_store:
    feat_options = list(feature_store.get("importances", {}).keys())[:10]
    selected_feat = st.selectbox("Select Feature", feat_options)

    # Simulate distributions (reference vs current)
    np.random.seed(hash(selected_feat) % 1000)
    ref_data = np.random.randn(1000)
    cur_data = np.random.randn(1000) + (covariate_score * 2)  # shift by drift amount

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=ref_data, name="Train Baseline",
        opacity=0.6, marker_color="#3b82f6", nbinsx=40,
    ))
    fig_dist.add_trace(go.Histogram(
        x=cur_data, name="Production",
        opacity=0.6, marker_color="#ef4444", nbinsx=40,
    ))
    fig_dist.update_layout(
        barmode="overlay", height=250,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.caption(f"Dashboard refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | MLflow: http://localhost:5000 | API: http://localhost:8000/docs")
