from fastapi import APIRouter, Request
from ml_model_packaging_service.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    predictor = request.app.state.predictor
    return HealthResponse(
        status="ok",
        xgb_model_path=predictor.xgb_model_path,
        lora_adapter_path=predictor.lora_adapter_path,
        threshold=predictor.threshold,
        shap_store_size=predictor.shap_store_size,
    )
