from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from ..config import TRAINING_CONFIG
from ..dataset import load_wine_data
from ..features import create_features_and_target


def main() -> None:
    model_path = Path("models") / "baseline_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Did you run training?"
        )

    model = joblib.load(model_path)

    df = load_wine_data(config=TRAINING_CONFIG)
    X, y = create_features_and_target(df, config=TRAINING_CONFIG)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TRAINING_CONFIG.test_size,
        random_state=TRAINING_CONFIG.random_state,
        stratify=y,
    )

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    detailed_metrics: Dict[str, Any] = {
        "accuracy": float(acc),
        "classification_report": cls_report,
        "confusion_matrix": cm,
    }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "metrics_detailed.json"

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(detailed_metrics, f, indent=2)

    print(f"Wrote detailed metrics to {metrics_path}")


if __name__ == "__main__":
    main()
