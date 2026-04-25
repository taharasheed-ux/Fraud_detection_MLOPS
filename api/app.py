"""
FastAPI Inference API
---------------------
Serves the deployed fraud detection model with:
  - POST /predict — fraud probability + label
  - GET  /health  — health check
  - GET  /metrics — Prometheus metrics endpoint

Exposes Prometheus counters/histograms for system and model monitoring.
Includes basic drift detection on incoming features.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import joblib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
)

logger = logging.getLogger(__name__)

# ---- Paths ----
DEPLOY_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "deployed_model")
PREPROC_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "preprocessing")
FEATURE_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "features")

# ---- Prometheus Metrics ----
REQUEST_COUNT = Counter(
    "prediction_request_total", "Total prediction requests",
    ["status"]
)
REQUEST_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total", "Total fraud predictions"
)
LEGIT_PREDICTIONS = Counter(
    "legit_predictions_total", "Total legitimate predictions"
)
FRAUD_RECALL = Gauge("fraud_recall", "Current fraud recall estimate")
FALSE_POSITIVE_RATE = Gauge("false_positive_rate", "Current false positive rate")
DATA_DRIFT_SCORE = Gauge("data_drift_score", "Current data drift score")
MODEL_CONFIDENCE = Histogram(
    "prediction_confidence", "Distribution of prediction confidence",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# ---- Global state ----
model = None
feature_columns = None
model_meta = None
scaler = None
prediction_history = []


# ---- Lifespan ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and artifacts on startup."""
    global model, feature_columns, model_meta, scaler

    model_path = os.path.join(DEPLOY_DIR, "model.joblib")
    features_path = os.path.join(DEPLOY_DIR, "feature_columns.joblib")
    meta_path = os.path.join(DEPLOY_DIR, "model_meta.joblib")
    scaler_path = os.path.join(PREPROC_DIR, "scaler.joblib")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)
    else:
        logger.warning("No deployed model found at %s", model_path)

    if os.path.exists(features_path):
        feature_columns = joblib.load(features_path)
        logger.info("Feature columns loaded: %d features", len(feature_columns))

    if os.path.exists(meta_path):
        model_meta = joblib.load(meta_path)
        logger.info("Model meta: %s", model_meta)

        # Initialize gauges with training metrics
        FRAUD_RECALL.set(model_meta.get("recall", 0))

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    yield

    logger.info("Shutting down API.")


# ---- App ----
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection with monitoring",
    version="1.0.0",
    lifespan=lifespan,
)


# ---- Schemas ----
class TransactionInput(BaseModel):
    """Input transaction data. Pass features as key-value pairs."""
    features: dict

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "TransactionAmt": 100.0,
                    "card1": 1234,
                    "ProductCD_freq": 0.3,
                    "hour": 14,
                    "dayofweek": 3,
                }
            }
        }


class PredictionOutput(BaseModel):
    fraud_probability: float
    is_fraud: bool
    confidence: float
    model_name: str | None


# ---- Drift detection helper ----
def compute_drift_score(features_df: pd.DataFrame) -> float:
    """
    Simple drift detection: compute the fraction of feature values
    that fall outside ±3 std of the expected range (0 mean, 1 std
    for scaled features).
    """
    if features_df.empty:
        return 0.0
    out_of_range = ((features_df.abs() > 3).sum().sum()
                    / max(features_df.size, 1))
    return round(float(out_of_range), 4)


# ---- Endpoints ----
@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "no_model",
        "model_name": model_meta.get("model_name") if model_meta else None,
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(transaction: TransactionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")

    start_time = time.time()
    try:
        # Build feature vector
        df = pd.DataFrame([transaction.features])

        # Align to expected columns
        if feature_columns:
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            df = df[feature_columns]

        # Replace NaN/inf
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Apply scaling if available
        if scaler and isinstance(scaler, dict):
            for col in df.columns:
                if col in scaler:
                    m = scaler[col]["mean"]
                    s = scaler[col]["scale"]
                    df[col] = ((df[col] - m) / s).astype("float32")

        # Predict
        prob = float(model.predict_proba(df)[:, 1][0])
        is_fraud = prob >= 0.5
        confidence = prob if is_fraud else 1 - prob

        # Record metrics
        REQUEST_COUNT.labels(status="success").inc()
        MODEL_CONFIDENCE.observe(confidence)
        if is_fraud:
            FRAUD_PREDICTIONS.inc()
        else:
            LEGIT_PREDICTIONS.inc()

        # Drift tracking
        drift = compute_drift_score(df)
        DATA_DRIFT_SCORE.set(drift)

        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)

        return PredictionOutput(
            fraud_probability=round(prob, 4),
            is_fraud=is_fraud,
            confidence=round(confidence, 4),
            model_name=model_meta.get("model_name") if model_meta else None,
        )

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        REQUEST_LATENCY.observe(time.time() - start_time)
        logger.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
