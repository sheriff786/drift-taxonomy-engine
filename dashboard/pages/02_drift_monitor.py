"""Drift Monitoring Dashboard Page — Enhanced with full taxonomy visualization."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

st.set_page_config(page_title="Drift Monitor", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .drift-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 500; }
    .badge-critical { background: #fecaca; color: #991b1b; }
    .badge-warning { background: #fef3c7; color: #92400e; }
    .badge-ok { background: #d1fae5; color: #065f46; }
    .score-card { background: #1e1e2e; border: 1px solid #3d3d5c; border-radius: 10px; padding: 16px; text-align: center; }
    .alert-row { padding: 10px 14px; border-left: 3px solid; border-radius: 4px; margin-bottom: 8px; background: rgba(255,255,255,0.02); }
</style>
""", unsafe_allow_html=True)

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"
API_BASE = "http://localhost:8000/api/v1"


def load_latest_report():
    reports_dir = ARTIFACTS_DIR / "reports"
    if reports_dir.exists():
        reports = sorted(reports_dir.glob("drift_report_*.json"), reverse=True)
        if reports:
            with open(reports[0]) as f:
                return json.load(f)
    return None


st.title("🎯 Drift Monitoring & Taxonomy Engine")
st.caption("Real-time drift detection with automated classification and response playbook")
st.markdown("---")

# --- Upload & Diagnose Section ---
upload_col, status_col = st.columns([2, 1])

with status_col:
    st.markdown("### 📡 System Status")
    try:
        resp = requests.get(f"{API_BASE}/drift/status", timeout=5)
        if resp.status_code == 200:
            status = resp.json()
            severity = status.get("current_severity", "none")
            badge_class = "badge-critical" if severity in ["high", "critical"] else "badge-warning" if severity == "medium" else "badge-ok"
            st.markdown(f'<span class="drift-badge {badge_class}">Drift: {status["current_drift_type"]}</span>', unsafe_allow_html=True)
            st.metric("Checks (24h)", status.get("checks_run_24h", 0))
            st.metric("Current Severity", severity.upper())
        else:
            st.info("No active drift status.")
    except requests.ConnectionError:
        st.warning("API offline — showing cached data")
        report = load_latest_report()
        if report:
            st.metric("Last Type", report.get("drift_type", "N/A"))
            st.metric("Last Severity", report.get("severity", "N/A"))

with upload_col:
    st.markdown("### 🔬 Run Manual Drift Diagnosis")
    st.info("Upload current production data CSV to run the full taxonomy engine pipeline.")

    uploaded_file = st.file_uploader("Upload current data (CSV)", type=["csv"])

    if uploaded_file is not None:
        current_df = pd.read_csv(uploaded_file)
        st.write(f"✅ Loaded **{len(current_df):,}** samples × **{len(current_df.columns)}** features")

        if st.button("🚀 Run Drift Taxonomy Engine", type="primary"):
            with st.spinner("Running drift taxonomy engine..."):
                samples = current_df.to_dict(orient="records")
                try:
                    resp = requests.post(
                        f"{API_BASE}/drift/diagnose",
                        json={"current_samples": samples},
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        st.session_state["drift_result"] = result
                    else:
                        st.error(f"Diagnosis failed: {resp.text}")
                except requests.ConnectionError:
                    st.error("API not running. Start: `uvicorn api.main:app --reload`")

# --- Display Results ---
result = st.session_state.get("drift_result") or load_latest_report()

if result:
    st.markdown("---")
    st.subheader("📊 Drift Diagnosis Results")

    # Score cards
    s1, s2, s3, s4, s5 = st.columns(5)
    scores = [
        ("Covariate", result.get("covariate_score", 0), "#3b82f6"),
        ("Concept", result.get("concept_score", 0), "#8b5cf6"),
        ("Pipeline", result.get("pipeline_score", 0), "#f59e0b"),
        ("Target", result.get("target_score", 0), "#22c55e"),
        ("Overall", result.get("overall_score", 0), "#ef4444"),
    ]
    for col, (label, score, color) in zip([s1, s2, s3, s4, s5], scores):
        badge_color = "#ef4444" if score >= 0.2 else "#f59e0b" if score >= 0.1 else "#22c55e"
        with col:
            st.markdown(f"""
            <div class="score-card">
                <div style="font-size:11px;color:#9ca3af;margin-bottom:6px">{label.upper()}</div>
                <div style="font-size:28px;font-weight:600;color:{badge_color}">{score:.3f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Radar chart of scores
    radar_col, details_col = st.columns(2)

    with radar_col:
        st.markdown("#### Drift Score Radar")
        categories = ["Covariate", "Concept", "Pipeline", "Target"]
        values = [result.get("covariate_score", 0), result.get("concept_score", 0),
                  result.get("pipeline_score", 0), result.get("target_score", 0)]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.2)",
            line=dict(color="#3b82f6", width=2),
            name="Current",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[0.2, 0.2, 0.2, 0.2, 0.2],
            theta=categories + [categories[0]],
            line=dict(color="#ef4444", width=1, dash="dash"),
            name="Threshold",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 0.5])),
            height=300, margin=dict(l=40, r=40, t=40, b=40),
            showlegend=True, legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with details_col:
        st.markdown("#### Diagnosis Summary")
        drift_type = result.get("drift_type", "none")
        severity = result.get("severity", "none")
        action = result.get("action", "monitor")
        urgency = result.get("urgency_hours", "N/A")

        st.markdown(f"""
        | Property | Value |
        |----------|-------|
        | **Drift Type** | `{drift_type}` |
        | **Severity** | `{severity}` |
        | **Recommended Action** | `{action}` |
        | **Urgency Window** | `{urgency}` hours |
        | **Drifted Features** | {len(result.get('drifted_features', []))} |
        """)

        if result.get("playbook"):
            with st.expander("📖 Response Playbook", expanded=True):
                playbook = result["playbook"]
                if isinstance(playbook, dict):
                    st.markdown(f"**Priority:** {playbook.get('priority', 'N/A')}")
                    for i, step in enumerate(playbook.get("steps", []), 1):
                        st.markdown(f"{i}. {step}")
                else:
                    st.json(playbook)

    # Drifted features heatmap
    if result.get("drifted_features"):
        st.markdown("---")
        st.subheader("🔥 Drifted Features")

        drifted = result["drifted_features"]
        feature_scores = result.get("feature_scores", {})

        if feature_scores:
            feat_df = pd.DataFrame([
                {"Feature": f, "Drift Score": s, "Status": "🔴 Drifted" if f in drifted else "🟢 Stable"}
                for f, s in feature_scores.items()
            ]).sort_values("Drift Score", ascending=False)

            fig_feat = px.bar(
                feat_df.head(15), x="Feature", y="Drift Score",
                color="Drift Score",
                color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
                range_color=[0, 0.3],
            )
            fig_feat.add_hline(y=0.1, line_dash="dash", line_color="#f59e0b", annotation_text="Warning")
            fig_feat.add_hline(y=0.2, line_dash="dash", line_color="#ef4444", annotation_text="Critical")
            fig_feat.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.write(", ".join(drifted))

    # Pipeline issues
    if result.get("pipeline_issues"):
        st.markdown("---")
        st.subheader("⚠️ Pipeline Issues Detected")
        for issue in result["pipeline_issues"]:
            level = "🔴" if issue.get("severity") == "high" else "🟡"
            st.markdown(f"{level} **{issue.get('issue_type', 'unknown')}**: {issue.get('detail', '')}")
else:
    st.markdown("---")
    st.info("No drift reports available. Upload data above or run the drift pipeline to generate results.")
