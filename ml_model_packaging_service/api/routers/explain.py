from fastapi import APIRouter, Request, HTTPException

from ml_model_packaging_service.api.schemas import ExplainResponse

router = APIRouter()


@router.get("/{fund_id}", response_model=ExplainResponse)
async def get_explanation(fund_id: int, request: Request):
    predictor = request.app.state.predictor
    top_features = predictor.explain(fund_id)
    if not top_features:
        raise HTTPException(status_code=404, detail=f"No SHAP data for fund_id={fund_id}")
    return ExplainResponse(fund_id=fund_id, top_features=top_features)
