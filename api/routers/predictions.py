"""Prediction endpoints for fraud scoring."""

from fastapi import APIRouter, HTTPException
import pandas as pd

from api.schemas.prediction import PredictionRequest, PredictionResponse
from api.dependencies import get_predictor

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Score a transaction for fraud probability."""
    try:
        predictor = get_predictor()

        # Convert request to DataFrame
        df = pd.DataFrame([sample.model_dump() for sample in request.samples])

        result = predictor.predict_with_metadata(df)

        return PredictionResponse(
            predictions=result["predictions"],
            probabilities=result["probabilities"],
            model_version=result["model_name"],
            n_samples=result["n_samples"],
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
