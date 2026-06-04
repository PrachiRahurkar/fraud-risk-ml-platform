from fastapi import APIRouter, Request

from ml_model_packaging_service.api.schemas import (
    FundFeatures, PredictionResponse, BatchPredictRequest, BatchPredictResponse
)

router = APIRouter()


@router.post("", response_model=PredictionResponse)
async def predict_single(features: FundFeatures, request: Request):
    predictor = request.app.state.predictor
    results = predictor.predict_batch([features.model_dump()])
    r = results[0]
    return PredictionResponse(**r)


@router.post("/batch", response_model=BatchPredictResponse)
async def predict_batch(body: BatchPredictRequest, request: Request):
    predictor = request.app.state.predictor
    records = [f.model_dump() for f in body.records]
    results = predictor.predict_batch(records)
    return BatchPredictResponse(predictions=[PredictionResponse(**r) for r in results])
