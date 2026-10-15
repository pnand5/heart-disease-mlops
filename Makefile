PY ?= .venv/bin/python

.PHONY: setup data train test lint run docker-build docker-run monitoring-up monitoring-down mlflow-ui

setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py && $(PY) -m src.data

train:
	$(PY) -m src.train

test:
	.venv/bin/pytest -v

lint:
	.venv/bin/ruff check .

run:
	.venv/bin/uvicorn api.app:app --reload

docker-build:
	docker build -t heart-disease-api:latest .

docker-run:
	docker run --rm -p 8000:8000 heart-disease-api:latest

monitoring-up:
	docker compose up -d --build

monitoring-down:
	docker compose down

mlflow-ui:
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
