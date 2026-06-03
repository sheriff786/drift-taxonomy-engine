"""Model management schemas."""

from pydantic import BaseModel
from typing import List, Dict, Optional


class ModelListResponse(BaseModel):
    """List of registered models."""
    models: List[str]
    total: int


class ModelInfoResponse(BaseModel):
    """Detailed model information."""
    name: str
    version: str
    metrics: Dict[str, float]
    registered_at: str
