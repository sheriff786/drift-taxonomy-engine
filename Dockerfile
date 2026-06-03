# Multi-stage Dockerfile for Drift Taxonomy Engine
# Stage 1: Base with dependencies
FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Application
FROM base AS app

WORKDIR /app

# Copy source code
COPY src/ src/
COPY api/ api/
COPY pipelines/ pipelines/
COPY dashboard/ dashboard/
COPY scripts/ scripts/
COPY configs/ configs/
COPY pyproject.toml .
COPY Makefile .

# Create artifact directories
RUN mkdir -p artifacts/models artifacts/reports artifacts/references data

# Default: run the API server
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
