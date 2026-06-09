"""
Ensemble predictor: loads XGBoost + optional LoRA adapter, runs inference.
"""
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from ml_evaluation_pipeline.explainability import load_shap_store
from ml_training_service.features.structured import engineer_features, TABULAR_FEATURE_COLS
from ml_training_service.features.text import format_prompt
from ml_training_service.data.preprocessing import NUMERIC_COLS, preprocess

logger = logging.getLogger(__name__)

LORA_MAX_LENGTH = int(os.getenv("LORA_MAX_LENGTH", "512"))
DEFAULT_NUMERIC_VALUES = {
    "category_id": 0.0,
    "goal": 0.0,
    "descr_len": 0.0,
    "title_len": 0.0,
    "identity_check_score": 0.0,
    "primary_email_address_checks__is_disposable": 0.0,
    "primary_email_address_checks__email_domain_creation_days": 0.0,
}


def _download_gcs_path(uri: str) -> str:
    from google.cloud import storage

    if not uri.startswith("gs://"):
        return uri

    bucket_name, _, blob_name = uri[5:].partition("/")
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    if blob_name.endswith("/") or not Path(blob_name).suffix:
        prefix = blob_name if blob_name.endswith("/") else f"{blob_name}/"
        local_dir = Path(tempfile.mkdtemp(prefix="fraud-risk-gcs-"))
        for blob in client.list_blobs(bucket_name, prefix=prefix):
            if blob.name.endswith("/"):
                continue
            rel_path = Path(blob.name).relative_to(prefix)
            dest = local_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(dest)
        return str(local_dir)

    suffix = Path(blob_name).suffix
    fd, local_path = tempfile.mkstemp(prefix="fraud-risk-gcs-", suffix=suffix)
    os.close(fd)
    bucket.blob(blob_name).download_to_filename(local_path)
    return local_path


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
        self._local_lora_adapter_path: Optional[str] = None

    def load(self) -> None:
        logger.info("Loading XGBoost model from %s", self.xgb_model_path)
        local_xgb_model_path = _download_gcs_path(self.xgb_model_path)
        self._xgb = xgb.XGBClassifier()
        self._xgb.load_model(local_xgb_model_path)

        local_shap_store_path = _download_gcs_path(self.shap_store_path)
        if Path(local_shap_store_path).exists():
            self._shap_store = load_shap_store(local_shap_store_path)
            logger.info("Loaded SHAP store: %d entries", len(self._shap_store))

    def load_lora_if_needed(self) -> None:
        if self._lora_model is not None or not self.lora_adapter_path:
            return

        local_adapter_path = _download_gcs_path(self.lora_adapter_path)
        if Path(local_adapter_path).exists():
            self._local_lora_adapter_path = local_adapter_path
            self._load_lora(local_adapter_path)

    def _load_lora(self, adapter_path: str) -> None:
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            cfg = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
            base_model_name = os.getenv("LORA_BASE_MODEL_PATH") or cfg["base_model_name_or_path"]
            logger.info("Loading LoRA adapter from %s (base: %s)", adapter_path, base_model_name)

            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            base = AutoModelForSequenceClassification.from_pretrained(
                base_model_name, num_labels=2, torch_dtype=dtype, device_map="auto",
            )
            base.config.pad_token_id = tokenizer.pad_token_id

            lora_model = PeftModel.from_pretrained(base, adapter_path)
            lora_model.eval()

            self._lora_model = lora_model
            self._lora_tokenizer = tokenizer
            logger.info("LoRA model ready (w_lora=%.2f)", self.w_lora)
        except Exception as e:
            self._lora_model = None
            self._lora_tokenizer = None
            logger.warning("Could not load LoRA adapter: %s — XGBoost only", e)

    def _lora_predict(self, records: list[dict]) -> np.ndarray:
        import torch
        texts = [
            format_prompt(r.get("title", "") or "", r.get("description", "") or "")
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

        for col in NUMERIC_COLS:
            default = DEFAULT_NUMERIC_VALUES.get(col, 0.0)
            if col not in df.columns:
                df[col] = default
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

        if "primary_phone_checks__line_type" not in df.columns:
            df["primary_phone_checks__line_type"] = "unknown"
        else:
            df["primary_phone_checks__line_type"] = df["primary_phone_checks__line_type"].fillna("unknown")

        df, self._num_imp, self._phone_enc = preprocess(
            df, self._num_imp, self._phone_enc, fit=(self._num_imp is None)
        )
        df = engineer_features(df)
        self._feature_names = [c for c in TABULAR_FEATURE_COLS if c in df.columns]
        return df[self._feature_names].values

    def predict_batch(self, records: list[dict]) -> list[dict]:
        X = self._to_feature_matrix(records)
        xgb_probs = self._xgb.predict_proba(X)[:, 1]
        requested_modes = [str(rec.get("model_mode") or "ensemble").lower() for rec in records]
        should_run_lora = (
            self.w_lora > 0
            and any(mode == "ensemble" for mode in requested_modes)
        )
        if should_run_lora:
            self.load_lora_if_needed()
            should_run_lora = self._lora_model is not None and self._lora_tokenizer is not None
        lora_probs = self._lora_predict(records) if should_run_lora else None

        results = []
        for i, rec in enumerate(records):
            fund_id = rec.get("fund_id")
            requested_mode = requested_modes[i]
            use_lora = requested_mode == "ensemble" and lora_probs is not None
            score = (
                float((1 - self.w_lora) * xgb_probs[i] + self.w_lora * lora_probs[i])
                if use_lora
                else float(xgb_probs[i])
            )
            results.append({
                "fund_id": fund_id,
                "fraud_score": score,
                "label": int(score >= self.threshold),
                "threshold": self.threshold,
                "model_mode": "ensemble" if use_lora else "xgb",
                "xgb_score": float(xgb_probs[i]),
                "lora_score": float(lora_probs[i]) if use_lora else None,
                "lora_weight": self.w_lora if use_lora else 0.0,
                "top_features": self._shap_store.get(str(fund_id), []),
            })
        return results

    def explain(self, fund_id: int) -> list[dict]:
        return self._shap_store.get(str(fund_id), [])

    @property
    def shap_store_size(self) -> int:
        return len(self._shap_store)

    @property
    def lora_loaded(self) -> bool:
        return self._lora_model is not None and self._lora_tokenizer is not None
