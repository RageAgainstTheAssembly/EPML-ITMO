# from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..clearml_utils import clearml_run
from ..config import TRAINING_CONFIG, TrainingConfig, load_model_config
from ..dataset import load_wine_data
from ..features import create_features_and_target
from ..mlflow_utils import training_run


def build_model_pipeline(model_params: Dict[str, Any] | None = None) -> Pipeline:
    if model_params is None:
        model_params = {}

    model_type = model_params.get("type", "logreg")
    random_state = model_params.get("random_state", TRAINING_CONFIG.random_state)

    if model_type == "logreg":
        clf = LogisticRegression(
            max_iter=model_params.get("max_iter", 1000),
            C=model_params.get("C", 1.0),
            multi_class=model_params.get("multi_class", "auto"),
            n_jobs=-1,
            random_state=random_state,
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=model_params.get("n_estimators", 100),
            max_depth=model_params.get("max_depth", None),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        clf = GradientBoostingClassifier(
            learning_rate=model_params.get("gb_learning_rate", 0.1),
            n_estimators=model_params.get("gb_n_estimators", 100),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unsupported model.type: {model_type}")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ]
    )


@clearml_run(
    default_task_name="train_wine_model",
    task_type=__import__("clearml").Task.TaskTypes.training,
    default_tags={"stage": "train"},
    register_model_name="wine_predictor_baseline",
)
@training_run(
    default_run_name="train_wine_model",
    default_tags={"project": "epml_itmo", "hw": "4", "stage": "train"},
)
def train_baseline(
    *,
    config: TrainingConfig = TRAINING_CONFIG,
    model_name: str = "logreg",
    override_model_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    model_cfg = load_model_config(model_name)
    base_model_params: Dict[str, Any] = model_cfg.get("model", {}) or {}
    model_params: Dict[str, Any] = base_model_params.copy()

    if override_model_params:
        model_params.update(override_model_params)

    df = load_wine_data(config=config)
    X, y = create_features_and_target(df, config=config)

    test_size = model_params.get("test_size", config.test_size)
    random_state = model_params.get("random_state", config.random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_model_pipeline(model_params=model_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {acc:.3f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "baseline_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved model to: {model_path}")

    metrics = {"accuracy": float(acc)}
    metrics_path = Path("metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    final_params: Dict[str, Any] = {
        **model_params,
        "test_size": test_size,
        "random_state": random_state,
    }

    return {
        "model": model,
        "params": final_params,
        "metrics": metrics,
        "artifacts": {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
        },
        "tags": {
            "stage": "train",
            "algorithm": model_params.get("type", model_name),
            "model_name": model_name,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="logreg",
        choices=["logreg", "random_forest", "gradient_boosting"],
        help="Which model config to use (configs/model/<name>.yaml)",
    )
    parser.add_argument(
        "--no-clearml",
        action="store_true",
        help="Disable ClearML logging for this run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_baseline(
        config=TRAINING_CONFIG,
        model_name=args.model,
        enable_clearml=not args.no_clearml,
    )
