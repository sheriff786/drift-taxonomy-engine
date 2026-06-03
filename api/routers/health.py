"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "service": "drift-taxonomy-engine"}


@router.get("/ready")
async def readiness_check():
    """Readiness check - verifies model and engine are loaded."""
    from api.dependencies import app_state

    model_loaded = "predictor" in app_state
    engine_loaded = "drift_engine" in app_state

    if model_loaded and engine_loaded:
        return {"status": "ready", "model_loaded": True, "engine_loaded": True}
    return {"status": "not_ready", "model_loaded": model_loaded, "engine_loaded": engine_loaded}
