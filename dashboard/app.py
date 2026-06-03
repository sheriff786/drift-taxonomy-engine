"""Drift Taxonomy Engine - Streamlit Dashboard."""

import streamlit as st

st.set_page_config(
    page_title="Drift Taxonomy Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Drift Taxonomy Engine Dashboard")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model Status", "Active", delta="Healthy")
with col2:
    st.metric("Current Drift", "None", delta="Stable")
with col3:
    st.metric("Last Check", "2 min ago")
with col4:
    st.metric("Alerts (24h)", "0", delta="0")

st.markdown("---")
st.markdown("""
### Navigation
Use the sidebar to navigate between pages:
- **Model Performance** - Track model metrics over time
- **Drift Monitor** - Real-time drift detection dashboard
- **Prediction Explorer** - Test predictions interactively
- **Action Playbook** - View recommended operational actions
""")
