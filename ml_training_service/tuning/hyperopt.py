"""
Optuna hyperparameter tuning for XGBoost.
Usage: conda activate fraud-risk && python -m ml_training_service.tuning.hyperopt
"""
import logging
import argparse

import mlflow
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np

from ml_training_service.features.structured import get_feature_matrix

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
    }

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in skf.split(X, y):
        model = xgb.XGBClassifier(**params)
        model.fit(X[train_idx], y[train_idx], verbose=False)
        probs = model.predict_proba(X[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], probs))

    return float(np.mean(aucs))


def run_study(
    train_df: pd.DataFrame,
    n_trials: int = 50,
    mlflow_tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "fraud-xgboost-tuning",
) -> dict:
    """Run Optuna study, log best params to MLflow, return best params dict."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    X = get_feature_matrix(train_df).values
    y = train_df["label"].values

    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials, n_jobs=1)

    best_params = study.best_params
    best_auc = study.best_value
    logger.info("Best AUC=%.4f | params=%s", best_auc, best_params)

    with mlflow.start_run(run_name="optuna-best"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_auc", best_auc)

    return best_params


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/train_data/fraud_data_train.csv")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--mlflow-uri", default="http://localhost:5000")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_path)
    run_study(train_df, n_trials=args.n_trials, mlflow_tracking_uri=args.mlflow_uri)
