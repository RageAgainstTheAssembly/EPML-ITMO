# from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from ..clearml_utils import clearml_run
from ..config import TRAINING_CONFIG
from ..dataset import load_wine_data
from ..features import create_features_and_target


def _extract_f1_macro(cls_report: Dict[str, Any]) -> Optional[float]:
    try:
        macro = cls_report.get("macro avg", {})
        val = macro.get("f1-score", None)
        return float(val) if val is not None else None
    except Exception:
        return None


@clearml_run(
    default_task_name="evaluate_wine_model",
    task_type=__import__("clearml").Task.TaskTypes.testing,
    default_tags={"stage": "evaluate"},
)
def evaluate_model(
    *,
    model_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(detailed_metrics, f, indent=2)

    print(f"Wrote detailed metrics to {output_path}")

    f1_macro = _extract_f1_macro(cls_report)

    return {
        "params": {
            "model_path": str(model_path),
            "output_path": str(output_path),
            "test_size": float(TRAINING_CONFIG.test_size),
            "random_state": int(TRAINING_CONFIG.random_state),
        },
        "metrics": {
            "accuracy": float(acc),
            **({"f1_macro": float(f1_macro)} if f1_macro is not None else {}),
        },
        "artifacts": {
            "metrics_detailed_path": str(output_path),
        },
        "tags": {
            "stage": "evaluate",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path("models") / "baseline_model.joblib"),
        help="Path to trained model .joblib",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(Path("reports") / "metrics_detailed.json"),
        help="Path to write detailed metrics JSON",
    )
    parser.add_argument(
        "--no-clearml",
        action="store_true",
        help="Disable ClearML logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(
        model_path=Path(args.model_path),
        output_path=Path(args.output_path),
        enable_clearml=not args.no_clearml,
    )


if __name__ == "__main__":
    main()
