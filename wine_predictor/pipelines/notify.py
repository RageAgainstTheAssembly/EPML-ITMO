from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    metrics_path = Path("reports") / "metrics_detailed.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Detailed metrics not found at {metrics_path}. "
            "Run the evaluation stage first."
        )

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    accuracy = metrics.get("accuracy", None)

    print("\n=== Pipeline notification ===")
    if accuracy is not None:
        print(f"Final test accuracy: {accuracy:.3f}")
    else:
        print("Could not find accuracy in detailed metrics.")

    print("Classification report keys:")
    cls_report = metrics.get("classification_report", {})
    print(", ".join(cls_report.keys()))

    print("DVC pipeline has finished successfully.\n")


if __name__ == "__main__":
    main()
