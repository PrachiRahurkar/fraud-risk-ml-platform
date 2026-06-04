"""
Bias and fairness evaluation: slice AUC by category_id and phone_line_type.
Flags slices where AUC drops > 5% below overall AUC.
"""
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

ALERT_DROP_THRESHOLD = 0.05


def slice_auc(
    df: pd.DataFrame,
    probs: np.ndarray,
    group_col: str,
) -> pd.DataFrame:
    """Compute AUC per slice of group_col. Returns DataFrame with slice stats."""
    df = df.copy()
    df["_prob"] = probs
    records = []
    for group_val, grp in df.groupby(group_col):
        if grp["label"].nunique() < 2:
            continue
        auc = roc_auc_score(grp["label"].values, grp["_prob"].values)
        records.append({
            "group": group_col,
            "value": group_val,
            "n": len(grp),
            "fraud_rate": float(grp["label"].mean()),
            "auc": float(auc),
        })
    return pd.DataFrame(records).sort_values("auc")


def evaluate_bias(
    df: pd.DataFrame,
    probs: np.ndarray,
    overall_auc: float,
    slice_cols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Run slice evaluation across all slice_cols.
    Logs a warning for any slice with AUC < overall_auc - ALERT_DROP_THRESHOLD.
    Returns dict of {col: slice_df}.
    """
    if slice_cols is None:
        slice_cols = ["category_id", "primary_phone_checks__line_type"]

    results = {}
    for col in slice_cols:
        if col not in df.columns:
            continue
        slice_df = slice_auc(df, probs, col)
        slice_df["auc_gap"] = overall_auc - slice_df["auc"]
        flagged = slice_df[slice_df["auc_gap"] > ALERT_DROP_THRESHOLD]
        if not flagged.empty:
            logger.warning(
                "Bias alert — %d slices of '%s' have AUC drop > %.0f%%:\n%s",
                len(flagged), col, ALERT_DROP_THRESHOLD * 100,
                flagged[["value", "n", "auc", "auc_gap"]].to_string(index=False)
            )
        results[col] = slice_df

    return results
