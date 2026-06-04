"""
gRPC MonitoringService: serves drift reports and rolling metrics.
Usage: conda activate fraud-risk && python ml-observability-monitoring/grpc_server.py
"""
import logging
import os
import time
from concurrent import futures

import grpc
from grpc_tools import protoc

# Generate Python stubs from proto if not already done
import subprocess, sys, pathlib

PROTO_DIR = pathlib.Path(__file__).parent / "proto"
STUB_DIR = pathlib.Path(__file__).parent

def ensure_stubs():
    if not (STUB_DIR / "monitoring_pb2.py").exists():
        protoc.main([
            "grpc_tools.protoc",
            f"-I{PROTO_DIR}",
            f"--python_out={STUB_DIR}",
            f"--grpc_python_out={STUB_DIR}",
            str(PROTO_DIR / "monitoring.proto"),
        ])

ensure_stubs()

import monitoring_pb2        # noqa: E402
import monitoring_pb2_grpc   # noqa: E402

from ml_observability_monitoring.drift import (
    load_reference, load_current, run_drift_report
)
from ml_observability_monitoring.performance import compute_rolling_metrics
from ml_observability_monitoring.alerting import check_drift_alert, check_performance_alert

logger = logging.getLogger(__name__)

REFERENCE_PATH = os.getenv("REFERENCE_PATH", "data/train_data/fraud_data_train.csv")
INFERENCE_LOG_PATH = os.getenv("INFERENCE_LOG_PATH", "data/inference_logs/predictions.jsonl")
FEEDBACK_LABEL_PATH = os.getenv("FEEDBACK_LABEL_PATH", "data/feedback/labels.jsonl")


class MonitoringServicer(monitoring_pb2_grpc.MonitoringServiceServicer):
    def GetDriftReport(self, request, context):
        try:
            ref_df = load_reference(REFERENCE_PATH)
            cur_df = load_current(INFERENCE_LOG_PATH)
            summary = run_drift_report(ref_df, cur_df)
            check_drift_alert(summary)
            return monitoring_pb2.DriftReport(
                drift_detected=summary["drift_detected"],
                drifted_features=summary["drifted_features"],
                share_of_drifted_features=summary["share_of_drifted_features"],
                generated_at=summary["generated_at"],
            )
        except Exception as e:
            logger.error("GetDriftReport error: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return monitoring_pb2.DriftReport()

    def GetRollingMetrics(self, request, context):
        try:
            window_days = 7
            if request.window.endswith("d"):
                window_days = int(request.window[:-1])
            metrics = compute_rolling_metrics(
                INFERENCE_LOG_PATH, FEEDBACK_LABEL_PATH, window_days
            )
            check_performance_alert(metrics)
            return monitoring_pb2.RollingMetrics(
                auc_roc=metrics.get("auc_roc") or 0.0,
                pr_auc=metrics.get("pr_auc") or 0.0,
                f1=metrics.get("f1") or 0.0,
                precision_at_100=metrics.get("precision_at_100") or 0.0,
                precision_at_500=metrics.get("precision_at_500") or 0.0,
                rolling_auc=metrics.get("rolling_auc") or 0.0,
                window=request.window,
                computed_at=metrics.get("computed_at", ""),
            )
        except Exception as e:
            logger.error("GetRollingMetrics error: %s", e)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return monitoring_pb2.RollingMetrics()


def serve(port: int = 50052) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    monitoring_pb2_grpc.add_MonitoringServiceServicer_to_server(MonitoringServicer(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info("MonitoringService gRPC server started on :%d", port)
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
