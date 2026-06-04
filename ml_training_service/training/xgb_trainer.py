"""
XGBoost training with Ray for distributed execution and MLflow tracking.
"""
import argparse
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from ml_training_service.data.preprocessing import preprocess
from ml_training_service.features.structured import engineer_features, get_feature_matrix

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 1.0,
    "random_state": 42,
    "tree_method": "hist",
    "n_jobs": -1,
}


def train_cv(
    df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    n_folds: int = 5,
    mlflow_tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "fraud-xgboost",
) -> tuple[xgb.XGBClassifier, float]:
    """Train XGBoost with stratified k-fold CV. Returns best model + mean AUC."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    params = {**DEFAULT_PARAMS, **(params or {})}
    X = get_feature_matrix(df).values
    y = df["label"].values

    fold_aucs = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    with mlflow.start_run():
        mlflow.log_params(params)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            val_probs = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, val_probs)
            fold_aucs.append(auc)
            logger.info("Fold %d AUC: %.4f", fold + 1, auc)

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        mlflow.log_metrics({"cv_auc_mean": mean_auc, "cv_auc_std": std_auc})
        logger.info("CV AUC: %.4f ± %.4f", mean_auc, std_auc)

        # Retrain on full data with best params
        final_model = xgb.XGBClassifier(**params)
        final_model.fit(X, y)
        mlflow.xgboost.log_model(final_model, artifact_path="xgb-model",
                                  registered_model_name="fraud-xgboost")

    return final_model, mean_auc


def train_and_save(
    train_df: pd.DataFrame,
    output_path: str = "models/xgb_model.json",
    params: dict[str, Any] | None = None,
    n_folds: int = 5,
    mlflow_tracking_uri: str = "http://localhost:5000",
) -> xgb.XGBClassifier:
    model, auc = train_cv(train_df, params=params,
                          n_folds=n_folds,
                          mlflow_tracking_uri=mlflow_tracking_uri)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)
    logger.info("XGBoost model saved to %s (CV AUC=%.4f)", output_path, auc)
    return model


def load_training_frame(train_path: str) -> pd.DataFrame:
    """Load raw fraud data and build model-ready structured features."""
    train_df = pd.read_csv(train_path)
    train_df, _, _ = preprocess(train_df, fit=True)
    return engineer_features(train_df)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train_data/fraud_data_train.csv")
    parser.add_argument("--output-path", default="models/xgb_model.json")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    train_df = load_training_frame(args.train_path)
    train_and_save(
        train_df,
        output_path=args.output_path,
        n_folds=args.n_folds,
        mlflow_tracking_uri=args.mlflow_uri,
    )


if __name__ == "__main__":
    main()
