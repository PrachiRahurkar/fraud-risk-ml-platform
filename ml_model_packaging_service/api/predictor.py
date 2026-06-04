"""
Ensemble predictor: loads XGBoost + optional LoRA adapter, runs inference.
"""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from ml_evaluation_pipeline.explainability import load_shap_store
from ml_training_service.features.structured import engineer_features, TABULAR_FEATURE_COLS
from ml_training_service.data.preprocessing import preprocess

logger = logging.getLogger(__name__)

LORA_MAX_LENGTH = 256


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
        self._lora_model = None
        self._lora_tokenizer = None

    def load(self) -> None:
        logger.info("Loading XGBoost model from %s", self.xgb_model_path)
        self._xgb = xgb.XGBClassifier()
        self._xgb.load_model(self.xgb_model_path)

        if Path(self.shap_store_path).exists():
            self._shap_store = load_shap_store(self.shap_store_path)
            logger.info("Loaded SHAP store: %d entries", len(self._shap_store))

        if self.lora_adapter_path and Path(self.lora_adapter_path).exists():
            self._load_lora(self.lora_adapter_path)

    def _load_lora(self, adapter_path: str) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            cfg = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
            base_model_name = cfg["base_model_name_or_path"]
            logger.info("Loading LoRA adapter from %s (base: %s)", adapter_path, base_model_name)

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            base = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=2, torch_dtype=dtype, device_map="auto",
            )
            self._lora_model = PeftModel.from_pretrained(base, adapter_path)
            self._lora_model.eval()

            self._lora_tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            if self._lora_tokenizer.pad_token is None:
                self._lora_tokenizer.pad_token = self._lora_tokenizer.eos_token

            logger.info("LoRA model ready (w_lora=%.2f)", self.w_lora)
        except Exception as e:
            logger.warning("Could not load LoRA adapter: %s — XGBoost only", e)

    def _lora_predict(self, records: list[dict]) -> np.ndarray:
        import torch
        texts = [
            f"Title: {r.get('title', '') or ''}\n\nDescription: {r.get('description', '') or ''}"
            for r in records
        ]
        inputs = self._lora_tokenizer(
            texts, truncation=True, padding=True,
            max_length=LORA_MAX_LENGTH, return_tensors="pt",
        ).to(self._lora_model.device)
        with torch.no_grad():
            logits = self._lora_model(**inputs).logits
        return torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

    def _to_feature_matrix(self, records: list[dict]) -> np.ndarray:
        df = pd.DataFrame(records)
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

        if self.w_lora > 0 and self._lora_model is not None:
            lora_probs = self._lora_predict(records)
            final_probs = (1 - self.w_lora) * xgb_probs + self.w_lora * lora_probs
        else:
            final_probs = xgb_probs

        results = []
        for i, rec in enumerate(records):
            fund_id = rec.get("fund_id")
            score = float(final_probs[i])
            results.append({
                "fund_id": fund_id,
                "fraud_score": score,
                "label": int(score >= self.threshold),
                "threshold": self.threshold,
                "top_features": self._shap_store.get(str(fund_id), []),
            })
        return results

    def explain(self, fund_id: int) -> list[dict]:
        return self._shap_store.get(str(fund_id), [])

    @property
    def shap_store_size(self) -> int:
        return len(self._shap_store)
