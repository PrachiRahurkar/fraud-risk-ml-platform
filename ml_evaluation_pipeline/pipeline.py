"""
Evaluation pipeline: load model → predict → metrics → SHAP → bias → log to MLflow.
Usage: conda activate fraud-risk && python -m ml_evaluation_pipeline.pipeline
"""
import argparse
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from ml_evaluation_pipeline.metrics import compute_metrics, log_metrics_to_mlflow
from ml_evaluation_pipeline.explainability import (
    compute_shap_values, global_importance, save_shap_store
)
from ml_evaluation_pipeline.bias import evaluate_bias
from ml_training_service.features.structured import get_feature_matrix, TABULAR_FEATURE_COLS
from ml_training_service.data.ingestion import load_and_merge
from ml_training_service.data.preprocessing import preprocess
from ml_training_service.features.structured import engineer_features

logger = logging.getLogger(__name__)


def run_evaluation(
    test_fraud_path: str = "data/test_data/fraud_data_test.csv",
    test_fund_path: str = "data/test_data/fund_data_test.csv",
    xgb_model_path: str = "models/xgb_model.json",
    shap_store_path: str = "models/shap_store.json",
    mlflow_tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "fraud-evaluation",
) -> None:
    logging.basicConfig(level=logging.INFO)
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load & preprocess test data
    test_df = load_and_merge(test_fraud_path, test_fund_path, "test")
    test_df, num_imp, phone_enc = preprocess(test_df, fit=True)
    test_df = engineer_features(test_df)

    X_test = get_feature_matrix(test_df).values
    y_test = test_df["label"].values
    feature_names = [c for c in TABULAR_FEATURE_COLS if c in test_df.columns]

    # Load XGBoost model
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_model_path)
    probs = xgb_model.predict_proba(X_test)[:, 1]

    with mlflow.start_run(run_name="evaluation"):
        # Metrics
        metrics = compute_metrics(y_test, probs)
        log_metrics_to_mlflow(metrics)
        logger.info("Test AUC-ROC: %.4f | PR-AUC: %.4f | F1: %.4f",
                    metrics.auc_roc, metrics.pr_auc, metrics.f1)

        # SHAP
        shap_values, _ = compute_shap_values(xgb_model, X_test, feature_names)
        importance_df = global_importance(shap_values, feature_names)
        logger.info("Top 5 features by SHAP:\n%s",
                    importance_df.head(5).to_string(index=False))

        importance_path = "models/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        # SHAP store for API
        fund_ids = test_df["fund_id"].values
        save_shap_store(fund_ids, shap_values, feature_names, shap_store_path)
        mlflow.log_artifact(shap_store_path)

        # Bias
        bias_results = evaluate_bias(test_df, probs, overall_auc=metrics.auc_roc)
        for col, slice_df in bias_results.items():
            path = f"models/bias_{col}.csv"
            slice_df.to_csv(path, index=False)
            mlflow.log_artifact(path)

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-fraud-path", default="data/test_data/fraud_data_test.csv")
    parser.add_argument("--test-fund-path", default="data/test_data/fund_data_test.csv")
    parser.add_argument("--xgb-model-path", default="models/xgb_model.json")
    parser.add_argument("--shap-store-path", default="models/shap_store.json")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    args = parser.parse_args()

    run_evaluation(
        args.test_fraud_path, args.test_fund_path,
        args.xgb_model_path, args.shap_store_path,
        args.mlflow_uri,
    )
