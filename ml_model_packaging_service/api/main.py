"""
FastAPI ML serving application.
Start: conda activate fraud-risk && uvicorn ml_model_packaging_service.api.main:app --port 8000
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ml_model_packaging_service.api.predictor import EnsemblePredictor
from ml_model_packaging_service.api.routers import predict, explain, health

logger = logging.getLogger(__name__)

XGB_MODEL_PATH = os.getenv("XGB_MODEL_PATH", "models/xgb_model.json")
SHAP_STORE_PATH = os.getenv("SHAP_STORE_PATH", "models/shap_store.json")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "")
THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
W_LORA = float(os.getenv("W_LORA", "0.0"))

predictor = EnsemblePredictor(
    xgb_model_path=XGB_MODEL_PATH,
    shap_store_path=SHAP_STORE_PATH,
    lora_adapter_path=LORA_ADAPTER_PATH or None,
    threshold=THRESHOLD,
    w_lora=W_LORA,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    logger.info("Models loaded — serving on /predict")
    yield


app = FastAPI(
    title="Fraud Risk ML API",
    description="Fraud detection inference: predict, batch predict, explain.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(predict.router, prefix="/predict", tags=["predict"])
app.include_router(explain.router, prefix="/explain", tags=["explain"])
app.include_router(health.router, tags=["health"])

app.state.predictor = predictor
