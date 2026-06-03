"""Prediction endpoints for fraud scoring."""

from fastapi import APIRouter, HTTPException
import pandas as pd

from api.schemas.prediction import PredictionRequest, PredictionResponse, AVAILABLE_MODELS
from src.config.constants import FEATURE_NAME_MAPPING, FEATURE_NAME_REVERSE
from src.models.predictor import ModelPredictor
from src.models.registry import ModelRegistry

router = APIRouter()


@router.get("/models/available")
async def list_available_models():
    """List all models available for prediction."""
    registry = ModelRegistry()
    registered = registry.list_models()
    return {
        "available_models": registered,
        "default": "random_forest",
    }


@router.get("/features/mapping")
async def get_feature_mapping():
    """Get the mapping between original PCA columns and domain feature names."""
    return {
        "mapping": FEATURE_NAME_MAPPING,
        "reverse": FEATURE_NAME_REVERSE,
        "feature_names": list(FEATURE_NAME_MAPPING.values()),
        "original_names": list(FEATURE_NAME_MAPPING.keys()),
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Score a transaction for fraud probability.

    Pass `model_name` to select which model to use:
    - random_forest (default, best performer)
    - xgboost
    - lightgbm
    - logistic_regression
    """
    try:
        model_name = request.model_name

        # Validate model name
        if model_name:
            registry = ModelRegistry()
            available = registry.list_models()
            if model_name not in available:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model_name}' not found. Available: {available}"
                )

        predictor = ModelPredictor(model_name=model_name)

        # Convert request to DataFrame using model input format (V1-V28)
        df = pd.DataFrame([sample.to_model_input() for sample in request.samples])

        result = predictor.predict_with_metadata(df)

        return PredictionResponse(
            predictions=result["predictions"],
            probabilities=result["probabilities"],
            model_name=result["model_used"],
            model_version=result["model_version"],
            n_samples=result["n_samples"],
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/explain")
async def explain_prediction(request: PredictionRequest):
    """
    Get SHAP-based explanations for predictions.

    Returns per-prediction feature contributions explaining WHY the model
    flagged a transaction as fraud or legitimate.
    """
    try:
        from src.explainability.shap_explainer import SHAPExplainer

        model_name = request.model_name

        # Validate model
        if model_name:
            registry = ModelRegistry()
            available = registry.list_models()
            if model_name not in available:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model_name}' not found. Available: {available}"
                )

        # Convert to model input format
        df = pd.DataFrame([sample.to_model_input() for sample in request.samples])

        # Generate explanations
        explainer = SHAPExplainer(model_name=model_name)
        explanations = explainer.explain(df, top_n=10)

        return {
            "explanations": explanations,
            "n_samples": len(explanations),
            "model_name": model_name or "default (best)",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
