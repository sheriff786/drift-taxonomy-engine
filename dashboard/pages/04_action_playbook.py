"""Action Playbook Page."""

import streamlit as st

st.set_page_config(page_title="Action Playbook", layout="wide")
st.title("Drift Response Playbook")

st.markdown("""
## Action Matrix

The Drift Taxonomy Engine maps drift type + severity to operational actions:
""")

import pandas as pd

action_matrix = pd.DataFrame({
    "Severity": ["Low", "Medium", "High", "Critical"],
    "Pipeline": ["Alert", "Investigate", "Block", "Block"],
    "Concept": ["Alert", "Incremental Retrain", "Full Retrain", "Full Retrain"],
    "Covariate": ["Monitor", "Alert", "Investigate", "Incremental Retrain"],
    "Target": ["Monitor", "Alert", "Incremental Retrain", "Full Retrain"],
    "Mixed": ["Investigate", "Incremental Retrain", "Full Retrain", "Full Retrain"],
}).set_index("Severity")

st.table(action_matrix)

st.markdown("""
---
## Response Time SLAs

| Action | Response Window |
|--------|----------------|
| Monitor | 1 week |
| Alert | 3 days |
| Investigate | 24 hours |
| Incremental Retrain | 12 hours |
| Full Retrain | 4 hours |
| Block | 1 hour (immediate) |

---
## Playbook Templates

### Block (Pipeline Critical)
1. **Block** - Halt prediction serving immediately
2. **Notify** - Page on-call engineer
3. **Diagnose** - Investigate data quality issues
4. **Fix** - Repair upstream pipeline
5. **Validate** - Re-run validation checks
6. **Resume** - Re-enable serving after fix

### Full Retrain (Concept Drift)
1. **Alert** - Notify ML team
2. **Analyze** - Investigate performance decay
3. **Collect** - Gather fresh labeled data
4. **Retrain** - Full retraining with hyperparameter search
5. **Validate** - Champion/challenger comparison
6. **Deploy** - Promote new model version
7. **Monitor** - Watch metrics for 48 hours

### Incremental Retrain (Covariate/Moderate)
1. **Alert** - Notify team
2. **Update** - Retrain with recent data
3. **Validate** - Regression tests
4. **Deploy** - Canary deploy
5. **Monitor** - Verify stabilization
""")
