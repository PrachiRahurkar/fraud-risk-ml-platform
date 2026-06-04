"""
Ensemble: weighted late fusion of LoRA text score and XGBoost tabular score.
Learns optimal weights on a held-out validation fold.
"""
import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def fuse_scores(
    lora_probs: np.ndarray,
    xgb_probs: np.ndarray,
    w_lora: float = 0.4,
) -> np.ndarray:
    """Weighted average: w_lora * lora + (1 - w_lora) * xgb."""
    return w_lora * lora_probs + (1.0 - w_lora) * xgb_probs


def learn_fusion_weight(
    lora_probs: np.ndarray,
    xgb_probs: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Find w_lora in [0,1] that maximises AUC on validation set."""
    def neg_auc(w):
        fused = fuse_scores(lora_probs, xgb_probs, w_lora=w)
        return -roc_auc_score(labels, fused)

    result = minimize_scalar(neg_auc, bounds=(0.0, 1.0), method="bounded")
    best_w = float(result.x)
    best_auc = -result.fun
    logger.info("Optimal w_lora=%.3f → AUC=%.4f", best_w, best_auc)
    return best_w


def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Youden-J threshold: maximises sensitivity + specificity - 1."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    threshold = float(thresholds[best_idx])
    logger.info("Optimal threshold: %.4f (TPR=%.3f, FPR=%.3f)",
                threshold, tpr[best_idx], fpr[best_idx])
    return threshold


class EnsemblePredictor:
    def __init__(self, w_lora: float = 0.4, threshold: float = 0.5):
        self.w_lora = w_lora
        self.threshold = threshold

    def fit(
        self,
        lora_probs: np.ndarray,
        xgb_probs: np.ndarray,
        labels: np.ndarray,
    ) -> "EnsemblePredictor":
        self.w_lora = learn_fusion_weight(lora_probs, xgb_probs, labels)
        fused = fuse_scores(lora_probs, xgb_probs, self.w_lora)
        self.threshold = find_optimal_threshold(fused, labels)
        return self

    def predict_proba(
        self, lora_probs: np.ndarray, xgb_probs: np.ndarray
    ) -> np.ndarray:
        return fuse_scores(lora_probs, xgb_probs, self.w_lora)

    def predict(
        self, lora_probs: np.ndarray, xgb_probs: np.ndarray
    ) -> np.ndarray:
        return (self.predict_proba(lora_probs, xgb_probs) >= self.threshold).astype(int)
