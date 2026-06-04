"""
Data drift detection using Evidently.
Compares training feature distribution vs recent inference log.
"""
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric

logger = logging.getLogger(__name__)

TABULAR_FEATURES = [
    "category_id", "goal", "descr_len", "title_len",
    "primary_phone_checks__line_type", "identity_check_score",
    "primary_email_address_checks__is_disposable",
    "primary_email_address_checks__email_domain_creation_days",
]


def load_reference(reference_path: str) -> pd.DataFrame:
    """Load training data as reference distribution."""
    df = pd.read_parquet(reference_path) if reference_path.endswith(".parquet") \
        else pd.read_csv(reference_path)
    available = [c for c in TABULAR_FEATURES if c in df.columns]
    return df[available].dropna()


def load_current(inference_log_path: str, n_recent: int = 1000) -> pd.DataFrame:
    """Load the N most recent inference log entries."""
    if inference_log_path.startswith("gs://"):
        from google.cloud import storage
        import io
        bucket, blob_path = inference_log_path[5:].split("/", 1)
        client = storage.Client()
        content = client.bucket(bucket).blob(blob_path).download_as_text()
        rows = [json.loads(line) for line in content.strip().splitlines()]
    else:
        with open(inference_log_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

    df = pd.DataFrame(rows[-n_recent:])
    available = [c for c in TABULAR_FEATURES if c in df.columns]
    return df[available].dropna()


def run_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str = "models/drift_report.html",
) -> dict:
    """
    Run Evidently drift report. Returns summary dict with drift_detected,
    drifted_features, share_of_drifted_features.
    """
    report = Report(metrics=[DataDriftPreset(), DatasetDriftMetric()])
    report.run(reference_data=reference_df, current_data=current_df)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(output_path)
    logger.info("Drift report saved to %s", output_path)

    result = report.as_dict()
    dataset_metric = next(
        (m for m in result["metrics"] if m["metric"] == "DatasetDriftMetric"), {}
    )
    result_data = dataset_metric.get("result", {})

    drifted_features = [
        feat for feat, stats in result.get("metrics", [{}])[0]
        .get("result", {}).get("drift_by_columns", {}).items()
        if stats.get("drift_detected")
    ] if result.get("metrics") else []

    return {
        "drift_detected": result_data.get("dataset_drift", False),
        "drifted_features": drifted_features,
        "share_of_drifted_features": result_data.get("share_of_drifted_columns", 0.0),
        "generated_at": datetime.utcnow().isoformat(),
    }
