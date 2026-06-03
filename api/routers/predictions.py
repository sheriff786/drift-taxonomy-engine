"""Prediction endpoints for fraud scoring."""

from fastapi import APIRouter, HTTPException
import pandas as pd

from api.schemas.prediction import PredictionRequest, PredictionResponse, AVAILABLE_MODELS
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

        # Convert request to DataFrame
        df = pd.DataFrame([sample.model_dump() for sample in request.samples])

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
