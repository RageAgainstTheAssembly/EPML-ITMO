from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

import mlflow

DEFAULT_EXPERIMENT_NAME = "wine_quality_hw3"
DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
DEFAULT_ARTIFACT_LOCATION = "mlruns/artifacts"


def configure_mlflow(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> None:
    """Configure MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@contextmanager
def mlflow_run(
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Any:
    """Context manager for a single MLflow run."""
    with mlflow.start_run(run_name=run_name, tags=tags):
        yield


def log_params(params: Dict[str, Any]) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_artifact(path: str) -> None:
    mlflow.log_artifact(path)


def log_model_sklearn(model: Any, artifact_path: str = "model") -> None:
    import mlflow.sklearn

    mlflow.sklearn.log_model(model, artifact_path)


def training_run(
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for wrapping a training function into an MLflow run.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            configure_mlflow()
            with mlflow_run(run_name=run_name or fn.__name__, tags=tags):
                result = fn(*args, **kwargs)  # type: ignore[call-arg]

                if isinstance(result, dict):
                    params = result.get("params")
                    if isinstance(params, dict):
                        log_params(params)

                    metrics = result.get("metrics")
                    if isinstance(metrics, dict):
                        log_metrics(metrics)

                    artifacts = result.get("artifacts")
                    if isinstance(artifacts, dict):
                        for _, path in artifacts.items():
                            log_artifact(str(path))

                    model = result.get("model")
                    if model is not None:
                        log_model_sklearn(model)

                return result

        return wrapper

    return decorator
