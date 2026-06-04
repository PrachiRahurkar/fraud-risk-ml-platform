"""
Ensemble predictor: loads XGBoost + optional LoRA adapter, runs inference.
"""
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from ml_evaluation_pipeline.explainability import load_shap_store, top_features_for_prediction
from ml_training_service.features.structured import engineer_features, TABULAR_FEATURE_COLS
from ml_training_service.data.preprocessing import preprocess

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    def __init__(
        self,
        xgb_model_path: str,
        shap_store_path: str,
        threshold: float = 0.5,
        w_lora: float = 0.0,
        lora_adapter_path: Optional[str] = None,
    ):
        self.xgb_model_path = xgb_model_path
        self.shap_store_path = shap_store_path
        self.threshold = threshold
        self.w_lora = w_lora
        self.lora_adapter_path = lora_adapter_path

        self._xgb: Optional[xgb.XGBClassifier] = None
        self._shap_store: dict = {}
        self._num_imp = None
        self._phone_enc = None
        self._feature_names: list[str] = []

    def load(self) -> None:
        logger.info("Loading XGBoost model from %s", self.xgb_model_path)
        self._xgb = xgb.XGBClassifier()
        self._xgb.load_model(self.xgb_model_path)

        if Path(self.shap_store_path).exists():
            self._shap_store = load_shap_store(self.shap_store_path)
            logger.info("Loaded SHAP store: %d entries", len(self._shap_store))

        if self.lora_adapter_path and Path(self.lora_adapter_path).exists():
            logger.info("LoRA adapter found at %s (w_lora=%.2f)", self.lora_adapter_path, self.w_lora)

    def _to_feature_matrix(self, records: list[dict]) -> np.ndarray:
        df = pd.DataFrame(records)
        # Derive length features from text when not explicitly provided
        if "descr_len" not in df.columns or df["descr_len"].isna().all():
            df["descr_len"] = df.get("description", pd.Series([""] * len(df))).fillna("").str.len().astype(float)
        if "title_len" not in df.columns or df["title_len"].isna().all():
            df["title_len"] = df.get("title", pd.Series([""] * len(df))).fillna("").str.len().astype(float)
        df, self._num_imp, self._phone_enc = preprocess(
            df, self._num_imp, self._phone_enc, fit=(self._num_imp is None)
        )
        df = engineer_features(df)
        self._feature_names = [c for c in TABULAR_FEATURE_COLS if c in df.columns]
        return df[self._feature_names].values

    def predict_batch(self, records: list[dict]) -> list[dict]:
        X = self._to_feature_matrix(records)
        xgb_probs = self._xgb.predict_proba(X)[:, 1]

        results = []
        for i, rec in enumerate(records):
            fund_id = rec.get("fund_id")
            score = float(xgb_probs[i])
            label = int(score >= self.threshold)
            top_features = self._shap_store.get(str(fund_id), [])
            results.append({
                "fund_id": fund_id,
                "fraud_score": score,
                "label": label,
                "threshold": self.threshold,
                "top_features": top_features,
            })
        return results

    def explain(self, fund_id: int) -> list[dict]:
        return self._shap_store.get(str(fund_id), [])

    @property
    def shap_store_size(self) -> int:
        return len(self._shap_store)
