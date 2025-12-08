from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import yaml  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import TRAINING_CONFIG, TrainingConfig
from ..dataset import load_wine_data
from ..features import create_features_and_target
from ..mlflow_utils import training_run


def load_params(path: str | Path = "params.yaml") -> Dict[str, Any]:
    param_path = Path(path)
    if not param_path.exists():
        return {}
    with param_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_logreg_pipeline(model_params: Dict[str, Any] | None = None) -> Pipeline:
    if model_params is None:
        model_params = {}

    max_iter = model_params.get("max_iter", 1000)
    C = model_params.get("C", 1.0)
    multi_class = model_params.get("multi_class", "auto")
    random_state = model_params.get("random_state", TRAINING_CONFIG.random_state)

    logreg = LogisticRegression(
        max_iter=max_iter,
        C=C,
        multi_class=multi_class,
        n_jobs=-1,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", logreg),
        ]
    )


@training_run(run_name="baseline_logreg")
def train_baseline(
    *,
    config: TrainingConfig = TRAINING_CONFIG,
) -> Dict[str, Any]:
    """Train a baseline logistic regression model"""
    params = load_params()
    model_params: Dict[str, Any] = params.get("model", {})

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

    model = build_logreg_pipeline(model_params=model_params)
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
    model_path = models_dir / "baseline_logreg.joblib"
    joblib.dump(model, model_path)
    print(f"\nSaved model to: {model_path}")

    metrics = {
        "accuracy": float(acc),
    }

    metrics_path = Path("metrics.json")
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")

    return {
        "model": model,
        "params": {
            **model_params,
            "test_size": test_size,
            "random_state": random_state,
            "model_type": "logreg",
        },
        "metrics": metrics,
        "artifacts": {
            "model_path": str(model_path),
            "metrics_path": str(metrics_path),
        },
    }


if __name__ == "__main__":
    train_baseline()
