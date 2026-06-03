"""Custom exception hierarchy."""


class DriftEngineError(Exception):
    """Base exception for drift taxonomy engine."""
    pass


class ModelNotFoundError(DriftEngineError):
    """Raised when a requested model is not in the registry."""
    pass


class ReferenceDataNotFoundError(DriftEngineError):
    """Raised when reference/baseline data is unavailable."""
    pass


class ValidationError(DriftEngineError):
    """Raised when data validation fails."""
    pass


class PipelineBlockedError(DriftEngineError):
    """Raised when drift severity requires blocking the pipeline."""
    pass
