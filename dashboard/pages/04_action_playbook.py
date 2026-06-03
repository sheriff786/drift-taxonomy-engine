"""Action Playbook Page — Enhanced with visual taxonomy and interactive responses."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path

st.set_page_config(page_title="Action Playbook", layout="wide")

st.markdown("""
<style>
    .playbook-step { background: #1e1e2e; border: 1px solid #3d3d5c; border-radius: 8px;
                     padding: 12px 16px; margin-bottom: 8px; border-left: 3px solid; }
    .step-block { border-left-color: #ef4444; }
    .step-retrain { border-left-color: #f59e0b; }
    .step-alert { border-left-color: #3b82f6; }
    .step-monitor { border-left-color: #22c55e; }
    .sla-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"

st.title("📖 Drift Response Playbook")
st.caption("Automated action mapping · Response time SLAs · Step-by-step runbooks")
st.markdown("---")

# --- Action Matrix Visualization ---
st.subheader("🎯 Action Decision Matrix")
st.caption("Drift Type × Severity → Recommended Action")

action_matrix = pd.DataFrame({
    "Severity": ["Low", "Medium", "High", "Critical"],
    "Pipeline": ["Alert", "Investigate", "Block", "Block"],
    "Concept": ["Alert", "Incremental Retrain", "Full Retrain", "Full Retrain"],
    "Covariate": ["Monitor", "Alert", "Investigate", "Incremental Retrain"],
    "Target": ["Monitor", "Alert", "Incremental Retrain", "Full Retrain"],
    "Mixed": ["Investigate", "Incremental Retrain", "Full Retrain", "Full Retrain"],
}).set_index("Severity")

# Heatmap visualization of action matrix
action_severity = {
    "Monitor": 1, "Alert": 2, "Investigate": 3,
    "Incremental Retrain": 4, "Full Retrain": 5, "Block": 6,
}
heatmap_data = action_matrix.map(lambda x: action_severity.get(x, 0))

fig_matrix = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=action_matrix.columns.tolist(),
    y=action_matrix.index.tolist(),
    text=action_matrix.values,
    texttemplate="%{text}",
    colorscale=[[0, "#d1fae5"], [0.3, "#fef3c7"], [0.6, "#fed7aa"], [0.8, "#fecaca"], [1, "#991b1b"]],
    showscale=False,
))
fig_matrix.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig_matrix, use_container_width=True)

# --- SLA Table ---
st.markdown("---")
st.subheader("⏱️ Response Time SLAs")

sla_data = pd.DataFrame({
    "Action": ["Monitor", "Alert", "Investigate", "Incremental Retrain", "Full Retrain", "Block"],
    "Response Window": ["1 week", "3 days", "24 hours", "12 hours", "4 hours", "1 hour (immediate)"],
    "Urgency": ["Low", "Low", "Medium", "High", "Critical", "Emergency"],
    "Automation Level": ["Fully Auto", "Auto + Notify", "Semi-Auto", "Human-in-Loop", "Human-in-Loop", "Auto Block + Page"],
})

st.dataframe(sla_data, use_container_width=True, hide_index=True)

# --- Current Recommendation ---
st.markdown("---")
st.subheader("🎯 Current Recommendation")

report = None
reports_dir = ARTIFACTS_DIR / "reports"
if reports_dir.exists():
    reports = sorted(reports_dir.glob("drift_report_*.json"), reverse=True)
    if reports:
        with open(reports[0]) as f:
            report = json.load(f)

if report:
    drift_type = report.get("drift_type", "none")
    severity = report.get("severity", "none")
    action = report.get("action", "monitor")

    col1, col2, col3 = st.columns(3)
    col1.metric("Detected Drift", drift_type.upper())
    col2.metric("Severity Level", severity.upper())
    col3.metric("Recommended Action", action.replace("_", " ").upper())

    st.markdown("")

    playbook = report.get("playbook")
    if playbook and isinstance(playbook, dict):
        st.markdown(f"**Priority:** {playbook.get('priority', 'N/A')}")
        st.markdown(f"**Response Window:** {playbook.get('response_window', 'N/A')}")
        st.markdown("")

        for i, step in enumerate(playbook.get("steps", []), 1):
            step_class = "step-block" if "block" in step.lower() else \
                         "step-retrain" if "retrain" in step.lower() else \
                         "step-alert" if "alert" in step.lower() or "notify" in step.lower() else "step-monitor"
            st.markdown(f"""
            <div class="playbook-step {step_class}">
                <strong>Step {i}:</strong> {step}
            </div>""", unsafe_allow_html=True)
else:
    st.success("✅ No active drift detected. System operating normally.")
    st.info("Run drift detection to generate actionable recommendations.")

# --- Playbook Templates ---
st.markdown("---")
st.subheader("📋 Playbook Templates")

tab_block, tab_retrain_full, tab_retrain_inc, tab_investigate = st.tabs(
    ["🔴 Block", "🟠 Full Retrain", "🟡 Incremental Retrain", "🔵 Investigate"]
)

with tab_block:
    st.markdown("""
    ### Block Response (Pipeline Critical)
    **Trigger:** Pipeline drift score > 0.5 OR data quality below threshold
    **SLA:** 1 hour (immediate halt)
    """)
    steps = [
        "🛑 **HALT** — Stop prediction serving immediately",
        "📟 **PAGE** — Notify on-call engineer via PagerDuty",
        "🔍 **DIAGNOSE** — Run pipeline validation checks",
        "🔧 **FIX** — Repair upstream data pipeline",
        "✅ **VALIDATE** — Re-run drift taxonomy engine",
        "🚀 **RESUME** — Re-enable serving after validation passes",
    ]
    for step in steps:
        st.markdown(f"  {step}")

with tab_retrain_full:
    st.markdown("""
    ### Full Retrain (Concept Drift)
    **Trigger:** Concept score > 0.2 with sustained performance decay
    **SLA:** 4 hours
    """)
    steps = [
        "🚨 **ALERT** — Notify ML team of performance degradation",
        "📊 **ANALYZE** — Run full concept drift analysis across segments",
        "📦 **COLLECT** — Gather fresh labeled data (minimum 1000 samples)",
        "🔄 **RETRAIN** — Full retraining with hyperparameter search (MLflow tracked)",
        "⚖️ **COMPARE** — Champion/challenger evaluation on holdout",
        "🚀 **DEPLOY** — Promote new model version to production",
        "👁️ **MONITOR** — Watch metrics for 48 hours post-deploy",
    ]
    for step in steps:
        st.markdown(f"  {step}")

with tab_retrain_inc:
    st.markdown("""
    ### Incremental Retrain (Covariate / Moderate)
    **Trigger:** Covariate drift with moderate severity
    **SLA:** 12 hours
    """)
    steps = [
        "📢 **ALERT** — Notify team via Slack/email",
        "📈 **UPDATE** — Retrain with recent data window (last 7 days)",
        "🧪 **VALIDATE** — Run regression tests + drift check",
        "🐦 **CANARY** — Deploy to 5% traffic canary",
        "📊 **VERIFY** — Confirm stabilization over 4 hours",
        "🚀 **PROMOTE** — Roll to full production",
    ]
    for step in steps:
        st.markdown(f"  {step}")

with tab_investigate:
    st.markdown("""
    ### Investigate (Warning Level)
    **Trigger:** Warning severity across any drift type
    **SLA:** 24 hours
    """)
    steps = [
        "🔔 **NOTIFY** — Send alert to ML team channel",
        "📊 **GATHER** — Collect drift evidence (feature distributions, scores)",
        "🔍 **ROOT CAUSE** — Identify drift source (upstream change, seasonal, real shift)",
        "📝 **DOCUMENT** — Record findings in drift log",
        "🎯 **DECIDE** — Determine if retraining needed or drift is transient",
    ]
    for step in steps:
        st.markdown(f"  {step}")
