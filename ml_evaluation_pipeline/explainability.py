"""
Model explainability: SHAP for XGBoost, per-prediction top-feature extraction.
"""
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, shap.TreeExplainer]:
    """Compute SHAP values for XGBoost model. Returns (shap_values, explainer)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    logger.info("SHAP values computed for %d samples", len(X))
    return shap_values, explainer


def global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Mean absolute SHAP — global feature importance ranking."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def top_features_for_prediction(
    shap_row: np.ndarray,
    feature_names: list[str],
    n: int = 3,
) -> list[dict]:
    """Return top-n features driving a single prediction."""
    idx = np.argsort(np.abs(shap_row))[::-1][:n]
    return [
        {
            "name": feature_names[i],
            "shap_value": float(shap_row[i]),
            "direction": "fraud" if shap_row[i] > 0 else "safe",
        }
        for i in idx
    ]


def save_shap_store(
    fund_ids: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: str = "models/shap_store.json",
) -> None:
    """Persist per-fund SHAP values as JSON for the inference API to serve."""
    store = {}
    for fund_id, row in zip(fund_ids, shap_values):
        store[str(int(fund_id))] = top_features_for_prediction(row, feature_names)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(store, f)
    logger.info("SHAP store saved: %d entries → %s", len(store), output_path)


def load_shap_store(path: str = "models/shap_store.json") -> dict[str, list[dict]]:
    with open(path) as f:
        return json.load(f)
