# from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..clearml_utils import clearml_run


def _safe_accuracy(metrics: Dict[str, Any]) -> Optional[float]:
    try:
        acc = metrics.get("accuracy", None)
        return float(acc) if acc is not None else None
    except Exception:
        return None


@clearml_run(
    default_task_name="pipeline_notify",
    task_type=__import__("clearml").Task.TaskTypes.inference,
    default_tags={"stage": "notify"},
)
def notify(
    *,
    metrics_path: Path,
) -> Dict[str, Any]:
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Detailed metrics not found at {metrics_path}. Run the evaluation stage first."
        )

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    accuracy = _safe_accuracy(metrics)

    print("\n=== Pipeline notification ===")
    if accuracy is not None:
        print(f"Final test accuracy: {accuracy:.3f}")
    else:
        print("Could not find accuracy in detailed metrics.")

    print("Classification report keys:")
    cls_report = metrics.get("classification_report", {})
    print(
        ", ".join(cls_report.keys()) if isinstance(cls_report, dict) else "<not a dict>"
    )

    print("DVC pipeline has finished successfully.\n")

    out: Dict[str, Any] = {
        "params": {"metrics_path": str(metrics_path)},
        "metrics": {},
        "artifacts": {"metrics_detailed_path": str(metrics_path)},
        "tags": {"stage": "notify"},
    }
    if accuracy is not None:
        out["metrics"]["accuracy"] = float(accuracy)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=str(Path("reports") / "metrics_detailed.json"),
        help="Path to metrics_detailed.json",
    )
    parser.add_argument(
        "--no-clearml",
        action="store_true",
        help="Disable ClearML logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notify(metrics_path=Path(args.metrics_path), enable_clearml=not args.no_clearml)


if __name__ == "__main__":
    main()
