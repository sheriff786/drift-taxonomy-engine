"""SHAP explainability endpoints."""

from fastapi import APIRouter, HTTPException
import pandas as pd

from api.schemas.prediction import TransactionSample
from src.explainability.shap_explainer import SHAPExplainer

router = APIRouter()


@router.post("/explain")
async def explain_prediction(
    samples: list[TransactionSample],
    model_name: str = None,
    top_n: int = 5,
):
    """
    Get SHAP-based feature explanations for predictions.

    Returns per-sample: prediction, probability, top contributing features
    with SHAP values indicating fraud/legitimate direction.
    """
    try:
        explainer = SHAPExplainer(model_name=model_name)
        df = pd.DataFrame([s.to_model_input() for s in samples])
        explanations = explainer.explain(df, top_n=top_n)
        return {
            "explanations": explanations,
            "model_name": explainer._model_name,
            "n_samples": len(explanations),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
