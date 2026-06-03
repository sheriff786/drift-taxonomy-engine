.PHONY: install train drift serve dashboard test lint clean

## Install dependencies
install:
	pip install -e ".[dev]"

## Run full training pipeline
train:
	python -m pipelines.training_pipeline

## Run drift check
drift:
	python scripts/run_drift_check.py

## Seed initial model and reference data
seed:
	python scripts/seed_data.py

## Start FastAPI server
serve:
	uvicorn api.main:app --reload --host localhost --port 8000

## Start Streamlit dashboard
dashboard:
	streamlit run dashboard/app.py

## Run tests
test:
	pytest tests/ -v --cov=src --cov-report=term-missing

## Run linter
lint:
	ruff check src/ api/ pipelines/
	black --check src/ api/ pipelines/

## Format code
format:
	black src/ api/ pipelines/ tests/
	ruff check --fix src/ api/ pipelines/

## Docker compose up
docker-up:
	docker-compose -f docker/docker-compose.yml up --build

## Docker compose down
docker-down:
	docker-compose -f docker/docker-compose.yml down

## Clean artifacts
clean:
	rm -rf artifacts/models/*.joblib
	rm -rf artifacts/reports/*.json
	find . -type d -name __pycache__ -exec rm -rf {} +
