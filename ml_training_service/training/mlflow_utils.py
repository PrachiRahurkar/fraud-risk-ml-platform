"""
MLflow helpers: experiment setup, model registration, stage promotion.
"""
import logging

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str, experiment_name: str) -> mlflow.MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return MlflowClient(tracking_uri=tracking_uri)


def get_best_run(experiment_name: str, metric: str = "auc_roc") -> dict:
    """Return the run with the highest value of `metric`."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")
    return runs[0]


def promote_model(
    model_name: str,
    run_id: str,
    stage: str = "Staging",
    artifact_path: str = "model",
) -> str:
    """Register a run's artifact to the Model Registry and set its stage."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"
    mv = mlflow.register_model(model_uri, model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    logger.info("Registered %s v%s → %s", model_name, mv.version, stage)
    return mv.version


def load_production_model(model_name: str, stage: str = "Production"):
    """Load the latest model version in a given stage from the registry."""
    model_uri = f"models:/{model_name}/{stage}"
    return mlflow.pyfunc.load_model(model_uri)
