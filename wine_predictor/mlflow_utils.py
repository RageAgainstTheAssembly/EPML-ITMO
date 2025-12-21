from __future__ import annotations

import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_EXPERIMENT_NAME = "wine_quality_hw4"
DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
DEFAULT_ARTIFACT_LOCATION = "mlruns/artifacts"


def configure_mlflow(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    artifact_location: Optional[str] = None,
) -> None:
    """
    Configure MLflow tracking and ensure the experiment exists.
    """
    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        if artifact_location:
            client.create_experiment(
                experiment_name, artifact_location=artifact_location
            )
        else:
            client.create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)


def _merge_tags(*tag_dicts: Optional[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for d in tag_dicts:
        if not d:
            continue
        for k, v in d.items():
            out[str(k)] = str(v)
    return out


def log_params(params: Dict[str, Any]) -> None:
    safe = {str(k): str(v) for k, v in (params or {}).items()}
    if safe:
        mlflow.log_params(safe)


def log_metrics(metrics: Dict[str, Any]) -> None:
    for k, v in (metrics or {}).items():
        val: float | None
        try:
            val = float(v)
        except (TypeError, ValueError):
            val = None

        if val is None:
            continue

        mlflow.log_metric(str(k), val)


def log_artifact(path: str | Path) -> None:
    p = Path(path)
    if p.exists() and p.is_file():
        mlflow.log_artifact(str(p))


def log_artifacts(paths: Iterable[str | Path]) -> None:
    for p in paths:
        log_artifact(p)


def log_model_sklearn(model: Any, name: str = "model") -> None:
    """
    Log a sklearn model to MLflow. Uses the newer `name=` API when available,
    falling back to older `artifact_path=` style for compatibility.
    """
    try:
        import mlflow.sklearn
    except Exception:
        return

    try:
        mlflow.sklearn.log_model(sk_model=model, name=name)  # type: ignore[call-arg]
    except TypeError:
        mlflow.sklearn.log_model(sk_model=model, artifact_path=name)  # type: ignore[call-arg]


@contextmanager
def mlflow_run(
    *,
    run_name: Optional[str],
    tags: Optional[Dict[str, Any]] = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    artifact_location: Optional[str] = None,
):
    configure_mlflow(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        artifact_location=artifact_location,
    )
    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(_merge_tags(tags))
        yield


def training_run(
    *,
    default_run_name: str = "run",
    default_tags: Optional[Dict[str, Any]] = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    artifact_location: Optional[str] = None,
    log_model: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for MLflow experiment tracking.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            run_name = kwargs.pop("run_name", None) or default_run_name
            mlflow_tags = kwargs.pop("mlflow_tags", None)

            enable_mlflow_kw = kwargs.pop("enable_mlflow", None)
            in_clearml = bool(os.getenv("CLEARML_TASK_ID"))
            enable_mlflow_effective = (
                bool(enable_mlflow_kw)
                if enable_mlflow_kw is not None
                else (not in_clearml)
            )

            if not enable_mlflow_effective:
                return fn(*args, **kwargs)

            tags = _merge_tags(default_tags, mlflow_tags)

            with mlflow_run(
                run_name=run_name,
                tags=tags,
                tracking_uri=tracking_uri,
                experiment_name=experiment_name,
                artifact_location=artifact_location,
            ):
                result = fn(*args, **kwargs)

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
                            ok = True
                            try:
                                log_artifact(str(path))
                            except Exception:
                                ok = False

                            if not ok:
                                continue

                    if log_model:
                        model = result.get("model")
                        if model is not None:
                            log_model_sklearn(model)

                    tags_from_result = result.get("tags")
                    if isinstance(tags_from_result, dict):
                        mlflow.set_tags(_merge_tags(tags_from_result))

                return result

        return wrapper

    return decorator
