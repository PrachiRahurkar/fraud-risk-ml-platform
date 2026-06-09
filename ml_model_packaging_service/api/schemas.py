from pydantic import BaseModel, Field
from typing import Literal, Optional


class FundFeatures(BaseModel):
    fund_id: int
    category_id: Optional[float] = None
    goal: Optional[float] = None
    descr_len: Optional[float] = None
    title_len: Optional[float] = None
    primary_phone_checks__line_type: Optional[str] = "unknown"
    identity_check_score: Optional[float] = None
    primary_email_address_checks__is_disposable: Optional[float] = 0.0
    primary_email_address_checks__email_domain_creation_days: Optional[float] = None
    title: Optional[str] = ""
    description: Optional[str] = ""
    model_mode: Literal["xgb", "ensemble"] = "ensemble"


class FeatureContribution(BaseModel):
    name: str
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    fund_id: int
    fraud_score: float = Field(ge=0.0, le=1.0)
    label: int = Field(ge=0, le=1)
    threshold: float
    model_mode: Literal["xgb", "ensemble"] = "xgb"
    xgb_score: Optional[float] = None
    lora_score: Optional[float] = None
    lora_weight: float = 0.0
    top_features: list[FeatureContribution]


class BatchPredictRequest(BaseModel):
    records: list[FundFeatures]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictionResponse]


class ExplainResponse(BaseModel):
    fund_id: int
    top_features: list[FeatureContribution]
    fraud_score: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    xgb_model_path: str
    lora_adapter_path: Optional[str]
    lora_loaded: bool
    lora_weight: float
    threshold: float
    shap_store_size: int
