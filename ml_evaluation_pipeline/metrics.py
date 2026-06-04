"""
Evaluation metrics: AUC-ROC, PR-AUC, F1, precision@k, threshold sweep.
"""
import logging
from dataclasses import dataclass, asdict

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    auc_roc: float
    pr_auc: float
    f1: float
    threshold: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision_at_100: float
    precision_at_500: float

    def to_dict(self) -> dict:
        return asdict(self)


def youden_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def precision_at_k(labels: np.ndarray, probs: np.ndarray, k: int) -> float:
    top_k_idx = np.argsort(probs)[::-1][:k]
    return float(labels[top_k_idx].mean())


def compute_metrics(
    labels: np.ndarray,
    probs: np.ndarray,
) -> EvalMetrics:
    threshold = youden_threshold(labels, probs)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    return EvalMetrics(
        auc_roc=float(roc_auc_score(labels, probs)),
        pr_auc=float(average_precision_score(labels, probs)),
        f1=float(f1_score(labels, preds, zero_division=0)),
        threshold=threshold,
        precision=float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        recall=float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        true_positives=int(tp),
        false_positives=int(fp),
        false_negatives=int(fn),
        true_negatives=int(tn),
        precision_at_100=precision_at_k(labels, probs, 100),
        precision_at_500=precision_at_k(labels, probs, 500),
    )


def log_metrics_to_mlflow(metrics: EvalMetrics, prefix: str = "test") -> None:
    mlflow.log_metrics({f"{prefix}_{k}": v for k, v in metrics.to_dict().items()
                        if isinstance(v, (int, float))})
