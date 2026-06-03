"""Model management endpoints."""

from fastapi import APIRouter, HTTPException

from api.schemas.model import ModelListResponse, ModelInfoResponse
from api.dependencies import get_predictor
from src.models.registry import ModelRegistry

router = APIRouter()


@router.get("/models", response_model=ModelListResponse)
async def list_models():
    """List all registered models."""
    registry = ModelRegistry()
    models = registry.list_models()
    return ModelListResponse(models=models, total=len(models))


@router.get("/models/{model_name}", response_model=ModelInfoResponse)
async def get_model_info(model_name: str):
    """Get metadata for a specific model."""
    registry = ModelRegistry()
    info = registry.get_model_info(model_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")
    return ModelInfoResponse(
        name=model_name,
        version=info.get("version", "unknown"),
        metrics=info.get("metrics", {}),
        registered_at=info.get("registered_at", "unknown"),
    )


@router.post("/models/retrain")
async def trigger_retrain():
    """Trigger model retraining pipeline."""
    # In production this would trigger an async job (Celery, Airflow, etc.)
    return {
        "status": "accepted",
        "message": "Retraining pipeline triggered. Check /models for updated version.",
    }
