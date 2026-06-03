"""Quick test of SHAP explainability module."""
import pandas as pd
import numpy as np
from src.explainability.shap_explainer import SHAPExplainer

# Create test transaction
np.random.seed(42)
test_data = pd.DataFrame([{f'V{i}': np.random.randn() for i in range(1, 29)} | {'Amount_scaled': 1.5, 'Time_scaled': -0.3}])

# Run SHAP explanation
explainer = SHAPExplainer('random_forest')
result = explainer.explain_single(test_data, top_n=5)

label = "FRAUD" if result["prediction"] == 1 else "LEGIT"
print(f"Prediction: {label}")
print(f"Probability: {result['probability']:.4f}")
print(f"Model: {result['model_used']}")
print(f"\nTop Contributors:")
for c in result['top_contributors']:
    direction = "-> FRAUD" if c['direction'] == 'fraud' else "-> LEGIT"
    print(f"  {c['feature']:30s} SHAP={c['shap_value']:+.4f} {direction}")
