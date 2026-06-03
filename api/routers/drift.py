"""Drift monitoring endpoints."""

from fastapi import APIRouter, HTTPException
import pandas as pd

from api.schemas.drift import DriftDiagnoseRequest, DriftDiagnoseResponse, DriftStatusResponse
from api.dependencies import get_drift_engine, get_ingestion

router = APIRouter()


@router.post("/drift/diagnose", response_model=DriftDiagnoseResponse)
async def diagnose_drift(request: DriftDiagnoseRequest):
    """Run drift diagnosis on provided current data vs reference."""
    try:
        engine = get_drift_engine()
        ingestion = get_ingestion()

        # Load reference data
        reference = ingestion.load_reference()

        # Convert current data from request
        current = pd.DataFrame([s.model_dump() for s in request.current_samples])

        diagnosis = engine.diagnose_quick(reference=reference, current=current)

        return DriftDiagnoseResponse(
            drift_type=diagnosis.drift_type,
            severity=diagnosis.severity,
            action=diagnosis.action,
            confidence=diagnosis.confidence,
            covariate_score=diagnosis.covariate_score,
            concept_score=diagnosis.concept_score,
            pipeline_score=diagnosis.pipeline_score,
            target_score=diagnosis.target_score,
            drifted_features=diagnosis.drifted_features,
            pipeline_issues=diagnosis.pipeline_issues,
            reasoning=diagnosis.reasoning,
            urgency_hours=diagnosis.urgency_hours,
            playbook=diagnosis.playbook,
            diagnosed_at=diagnosis.diagnosed_at,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Reference data not available: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift diagnosis failed: {str(e)}")


@router.get("/drift/status", response_model=DriftStatusResponse)
async def drift_status():
    """Get latest drift monitoring status."""
    # This would typically read from a monitoring store
    return DriftStatusResponse(
        last_check=None,
        current_drift_type="none",
        current_severity="none",
        checks_run_24h=0,
        alerts_triggered=0,
    )
