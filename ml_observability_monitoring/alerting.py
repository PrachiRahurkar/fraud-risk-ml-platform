"""
Threshold-based alerting: log warnings when drift or AUC drop detected.
Optionally posts to a webhook (Slack, PagerDuty, etc.).
"""
import logging
import os
import json
from datetime import datetime
from typing import Optional

import requests

logger = logging.getLogger(__name__)

WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "")
DRIFT_P_THRESHOLD = float(os.getenv("DRIFT_P_THRESHOLD", "0.05"))
AUC_DROP_THRESHOLD = float(os.getenv("AUC_DROP_THRESHOLD", "0.03"))
BASELINE_AUC = float(os.getenv("BASELINE_AUC", "0.82"))


def _send_webhook(message: str) -> None:
    if not WEBHOOK_URL:
        return
    try:
        requests.post(WEBHOOK_URL, json={"text": message}, timeout=5)
    except Exception as e:
        logger.warning("Webhook delivery failed: %s", e)


def check_drift_alert(drift_summary: dict) -> None:
    if drift_summary.get("drift_detected"):
        n = len(drift_summary.get("drifted_features", []))
        msg = (
            f"[FRAUD RISK] Data drift detected at {datetime.utcnow().isoformat()}. "
            f"{n} feature(s) drifted: {', '.join(drift_summary.get('drifted_features', []))}"
        )
        logger.warning(msg)
        _send_webhook(msg)


def check_performance_alert(rolling_metrics: dict, baseline_auc: float = BASELINE_AUC) -> None:
    current_auc = rolling_metrics.get("rolling_auc")
    if current_auc is None:
        return

    drop = baseline_auc - current_auc
    if drop > AUC_DROP_THRESHOLD:
        msg = (
            f"[FRAUD RISK] Model AUC drop alert at {datetime.utcnow().isoformat()}. "
            f"Baseline AUC={baseline_auc:.4f}, current={current_auc:.4f}, drop={drop:.4f}"
        )
        logger.warning(msg)
        _send_webhook(msg)
