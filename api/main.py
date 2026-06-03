"""FastAPI application factory and startup."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routers import predictions, drift, models, health, explain
from api.dependencies import load_resources, app_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and engine on startup, cleanup on shutdown."""
    load_resources()
    yield
    # Cleanup if needed
    app_state.clear()


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="Drift Taxonomy Engine API",
        description="Fraud detection with intelligent drift monitoring and operational actions.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(drift.router, prefix="/api/v1", tags=["Drift"])
    app.include_router(models.router, prefix="/api/v1", tags=["Models"])
    app.include_router(explain.router, prefix="/api/v1", tags=["Explainability"])

    return app


app = create_app()
