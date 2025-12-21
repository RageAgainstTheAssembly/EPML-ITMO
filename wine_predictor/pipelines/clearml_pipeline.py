import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from clearml import PipelineController

from wine_predictor.clearml_utils import (
    DEFAULT_CLEARML_PROJECT,
    _filter_kwargs_for_callable,
)

logger = logging.getLogger(__name__)


def _step_train(
    model_name: str, override_model_params: Optional[Dict[str, Any]] = None
) -> str:
    """
    Pipeline step: train model and return a path to the trained model file.
    Runs inside a ClearML pipeline step subprocess.
    """
    from wine_predictor.config import load_training_config
    from wine_predictor.modeling import train as train_mod

    fn = getattr(train_mod, "train_baseline", None)
    if fn is None:
        raise RuntimeError("wine_predictor.modeling.train.train_baseline was not found")

    config = load_training_config()

    models_dir = Path(config.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    default_model_path = models_dir / "baseline_model.joblib"

    kwargs: Dict[str, Any] = {
        "config": config,
        "model_name": model_name,
        "override_model_params": override_model_params,
        "enable_mlflow": False,
        "enable_clearml": True,
    }
    kwargs = _filter_kwargs_for_callable(fn, kwargs)

    result = fn(**kwargs)

    if isinstance(result, dict):
        artifacts = result.get("artifacts", {}) or {}
        mp = artifacts.get("model_path")
        if mp:
            return str(mp)

    return str(default_model_path)


def _step_evaluate(model_path: str) -> str:
    """
    Pipeline step: evaluate model at `model_path` and return a metrics json path.
    """
    from wine_predictor.pipelines import evaluate as eval_mod

    fn = getattr(eval_mod, "evaluate_model", None)
    if fn is None:
        raise RuntimeError(
            "wine_predictor.pipelines.evaluate.evaluate_model was not found"
        )

    output_path = Path("reports") / "metrics_detailed.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kwargs: Dict[str, Any] = {
        "model_path": Path(model_path),
        "output_path": output_path,
        "enable_mlflow": False,
        "enable_clearml": True,
    }
    kwargs = _filter_kwargs_for_callable(fn, kwargs)

    fn(**kwargs)
    return str(output_path)


def _step_notify(metrics_path: str) -> None:
    """
    Pipeline step: send/print a notification based on metrics file.
    """
    from wine_predictor.pipelines import notify as notify_mod

    fn = getattr(notify_mod, "notify", None)
    if fn is None:
        raise RuntimeError("wine_predictor.pipelines.notify.notify was not found")

    p = Path(metrics_path)
    if not p.exists():
        raise FileNotFoundError(f"metrics_path does not exist: {metrics_path}")

    kwargs: Dict[str, Any] = {
        "metrics_path": p,
        "enable_mlflow": False,
        "enable_clearml": True,
    }
    kwargs = _filter_kwargs_for_callable(fn, kwargs)

    fn(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ClearML pipeline: train -> evaluate -> notify"
    )
    p.add_argument(
        "--model", type=str, default="logreg", help="Model name (e.g., logreg)"
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    pipe = PipelineController(
        name="wine_predictor_hw5_pipeline",
        project=DEFAULT_CLEARML_PROJECT,
        version="1.0.0",
        add_pipeline_tags=True,
    )

    pipe.add_parameter("model_name", args.model)

    pipe.add_function_step(
        name="step_train",
        function=_step_train,
        function_kwargs={
            "model_name": "${pipeline.model_name}",
            "override_model_params": None,
        },
        function_return=["model_path"],
        cache_executed_step=False,
    )

    pipe.add_function_step(
        name="step_evaluate",
        function=_step_evaluate,
        function_kwargs={
            "model_path": "${step_train.model_path}",
        },
        function_return=["metrics_path"],
        parents=["step_train"],
        cache_executed_step=False,
    )

    pipe.add_function_step(
        name="step_notify",
        function=_step_notify,
        function_kwargs={"metrics_path": "${step_evaluate.metrics_path}"},
        parents=["step_evaluate"],
        cache_executed_step=False,
    )

    try:
        pipe.start_locally(run_pipeline_steps_locally=True)
    finally:
        try:
            pipe.stop()
        except Exception:
            logger.debug(
                "ClearML pipeline: pipe.stop() failed during cleanup", exc_info=True
            )


if __name__ == "__main__":
    main()
