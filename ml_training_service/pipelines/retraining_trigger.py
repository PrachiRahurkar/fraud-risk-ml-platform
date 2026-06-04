"""
Retraining trigger: watch the Kafka labels topic; kick off retraining when
≥ N new human-review labels have accumulated since the last training run.
"""
import json
import logging
import time
import argparse
import subprocess
from collections import deque
from datetime import datetime

from kafka import KafkaConsumer

logger = logging.getLogger(__name__)

LABELS_TOPIC = "fraud-risk.labels"


def consume_labels(
    bootstrap_servers: str,
    retrain_threshold: int,
    pipeline_cmd: list[str],
    check_interval_s: int = 60,
) -> None:
    consumer = KafkaConsumer(
        LABELS_TOPIC,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        auto_offset_reset="latest",
        group_id="retraining-trigger",
        enable_auto_commit=True,
    )

    pending: deque[dict] = deque()
    logger.info("Watching '%s' for new labels (threshold=%d)", LABELS_TOPIC, retrain_threshold)

    for message in consumer:
        label_record = message.value
        pending.append(label_record)
        logger.debug("Received label for fund_id=%s", label_record.get("fund_id"))

        if len(pending) >= retrain_threshold:
            logger.info("%d new labels accumulated — triggering retraining", len(pending))
            _trigger_retraining(pipeline_cmd)
            pending.clear()


def _trigger_retraining(cmd: list[str]) -> None:
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("Retraining completed successfully")
    else:
        logger.error("Retraining failed:\n%s", result.stderr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--retrain-threshold", type=int, default=100)
    args = parser.parse_args()

    pipeline_cmd = [
        "python", "-m", "ml_training_service.pipelines.training_pipeline",
        "--skip-lora",
    ]
    consume_labels(args.bootstrap_servers, args.retrain_threshold, pipeline_cmd)
