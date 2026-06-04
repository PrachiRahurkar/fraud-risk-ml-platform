"""
End-to-end training pipeline:
  ingest → preprocess → features → XGB train → LoRA train → ensemble → eval → register
Usage: conda activate fraud-risk && python -m ml_training_service.pipelines.training_pipeline
"""
import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from ml_training_service.data.ingestion import load_all
from ml_training_service.data.preprocessing import preprocess
from ml_training_service.features.structured import engineer_features, get_feature_matrix
from ml_training_service.features.text import build_text_dataset
from ml_training_service.features.store import save_features
from ml_training_service.training.xgb_trainer import train_and_save
from ml_training_service.training.ensemble import EnsemblePredictor
from ml_training_service.training.mlflow_utils import setup_mlflow, promote_model

logger = logging.getLogger(__name__)


def run_pipeline(
    data_root: str = "data",
    feature_store_path: str = "data/feature_store",
    model_output_dir: str = "models",
    mlflow_tracking_uri: str = "http://localhost:5000",
    skip_lora: bool = False,
    use_gcs: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO)
    setup_mlflow(mlflow_tracking_uri, "fraud-risk-pipeline")
    Path(model_output_dir).mkdir(parents=True, exist_ok=True)

    # --- Ingest ---
    logger.info("Step 1: Loading data")
    datasets = load_all(data_root, use_gcs=use_gcs)
    train_df = datasets["train"]
    test_df = datasets["test"]
    t2_train = datasets["technique2_train"]
    t2_test = datasets["technique2_test"]

    # --- Preprocess ---
    logger.info("Step 2: Preprocessing")
    train_df, num_imp, phone_enc = preprocess(train_df, fit=True)
    test_df, _, _ = preprocess(test_df, num_imp, phone_enc, fit=False)
    t2_train, _, _ = preprocess(t2_train, num_imp, phone_enc, fit=False)
    t2_test, _, _ = preprocess(t2_test, num_imp, phone_enc, fit=False)

    # --- Feature engineering ---
    logger.info("Step 3: Feature engineering")
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    t2_train = engineer_features(t2_train)
    t2_test = engineer_features(t2_test)

    save_features(train_df, feature_store_path, split="train")
    save_features(test_df, feature_store_path, split="test")

    # --- XGBoost training ---
    logger.info("Step 4: Training XGBoost")
    xgb_model = train_and_save(
        train_df,
        output_path=f"{model_output_dir}/xgb_model.json",
        mlflow_tracking_uri=mlflow_tracking_uri,
    )
    X_test = get_feature_matrix(test_df).values
    xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]

    # --- LoRA (optional — skipped when running locally without GPU) ---
    lora_test_probs = None
    if not skip_lora:
        logger.info("Step 5: Training LoRA (requires GPU)")
        from ml_training_service.training.lora_trainer import train as lora_train
        train_records = build_text_dataset(t2_train)
        eval_records = build_text_dataset(t2_test)
        lora_train(
            train_records, eval_records,
            output_dir=f"{model_output_dir}/lora-adapter",
            mlflow_tracking_uri=mlflow_tracking_uri,
        )
        logger.info("LoRA training complete. Adapter saved.")
    else:
        logger.info("Step 5: Skipping LoRA (--skip-lora flag set)")

    # --- Ensemble ---
    if lora_test_probs is not None:
        logger.info("Step 6: Fitting ensemble weights")
        X_val = get_feature_matrix(test_df).values
        xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
        ensemble = EnsemblePredictor()
        ensemble.fit(lora_test_probs, xgb_val_probs, test_df["label"].values)
        logger.info("Ensemble: w_lora=%.3f, threshold=%.4f",
                    ensemble.w_lora, ensemble.threshold)
    else:
        logger.info("Step 6: Skipping ensemble (no LoRA probs available)")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--feature-store-path", default="data/feature_store")
    parser.add_argument("--model-output-dir", default="models")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--use-gcs", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        data_root=args.data_root,
        feature_store_path=args.feature_store_path,
        model_output_dir=args.model_output_dir,
        mlflow_tracking_uri=args.mlflow_uri,
        skip_lora=args.skip_lora,
        use_gcs=args.use_gcs,
    )
