"""
Rolling model performance from inference logs + ground-truth feedback labels.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from ml_evaluation_pipeline.metrics import precision_at_k

logger = logging.getLogger(__name__)


def load_inference_logs(log_path: str) -> pd.DataFrame:
    rows = []
    if Path(log_path).exists():
        with open(log_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_feedback_labels(label_path: str) -> pd.DataFrame:
    rows = []
    if Path(label_path).exists():
        with open(label_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def compute_rolling_metrics(
    log_path: str = "data/inference_logs/predictions.jsonl",
    label_path: str = "data/feedback/labels.jsonl",
    window_days: int = 7,
) -> dict:
    """
    Join inference logs with feedback labels; compute AUC over the rolling window.
    Returns metric dict (empty values if no labeled data available).
    """
    logs_df = load_inference_logs(log_path)
    labels_df = load_feedback_labels(label_path)

    if logs_df.empty or labels_df.empty:
        logger.warning("Insufficient data for rolling metrics (no logs or labels yet)")
        return {
            "auc_roc": None, "pr_auc": None, "f1": None,
            "precision_at_100": None, "precision_at_500": None,
            "rolling_auc": None, "n_labeled": 0,
            "computed_at": datetime.utcnow().isoformat(),
        }

    cutoff = datetime.utcnow() - timedelta(days=window_days)

    logs_df["timestamp"] = pd.to_datetime(logs_df.get("timestamp", datetime.utcnow()))
    recent = logs_df[logs_df["timestamp"] >= cutoff]

    joined = recent.merge(
        labels_df[["fund_id", "is_fraud"]].rename(columns={"is_fraud": "true_label"}),
        on="fund_id", how="inner"
    )

    if len(joined) < 10:
        logger.warning("Only %d labeled examples in window — metrics may be unstable", len(joined))

    probs = joined["fraud_score"].values
    labels = joined["true_label"].astype(int).values
    preds = (probs >= 0.5).astype(int)

    return {
        "auc_roc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else None,
        "pr_auc": float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else None,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision_at_100": precision_at_k(labels, probs, 100) if len(labels) >= 100 else None,
        "precision_at_500": precision_at_k(labels, probs, 500) if len(labels) >= 500 else None,
        "rolling_auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else None,
        "n_labeled": int(len(joined)),
        "computed_at": datetime.utcnow().isoformat(),
    }
