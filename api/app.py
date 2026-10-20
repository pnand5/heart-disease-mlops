"""FastAPI service exposing the heart-disease model.

Run locally: uvicorn api.app:app --reload
"""
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from api.schemas import PatientFeatures, PredictionResponse
from src.predict import DEFAULT_MODEL_PATH, load_model, predict

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart-api")

MODEL_PATH = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

REQUESTS = Counter("api_requests_total", "HTTP requests", ["method", "path", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["path"])
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["diagnosis"])
ERRORS = Counter("model_prediction_errors_total", "Failed predictions")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(MODEL_PATH)
    logger.info("model loaded from %s", MODEL_PATH)
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(path).observe(elapsed)
        logger.info("%s %s -> %s in %.1f ms", request.method, path, status, elapsed * 1000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(patient: PatientFeatures):
    try:
        result = predict([patient.model_dump()], model=load_model(MODEL_PATH))[0]
    except Exception:
        ERRORS.inc()
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail="prediction failed")
    PREDICTIONS.labels(result["diagnosis"]).inc()
    logger.info("input=%s result=%s", patient.model_dump(), result)
    return result
