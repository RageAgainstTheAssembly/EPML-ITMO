from __future__ import annotations

from typing import Any, Dict, List

from .config import TRAINING_CONFIG
from .modeling.train import train_baseline


def build_experiment_grid() -> List[Dict[str, Any]]:
    experiments: List[Dict[str, Any]] = []

    for C in [0.1, 1.0, 10.0]:
        for test_size in [0.2, 0.3]:
            experiments.append(
                {
                    "model_name": "logreg",
                    "override_params": {
                        "type": "logreg",
                        "C": C,
                        "max_iter": 1000,
                        "test_size": test_size,
                    },
                }
            )

    for n_estimators in [50, 100, 200]:
        for max_depth in [None, 5]:
            experiments.append(
                {
                    "model_name": "random_forest",
                    "override_params": {
                        "type": "random_forest",
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                    },
                }
            )

    for lr in [0.05, 0.1, 0.2]:
        experiments.append(
            {
                "model_name": "gradient_boosting",
                "override_params": {
                    "type": "gradient_boosting",
                    "gb_learning_rate": lr,
                    "gb_n_estimators": 100,
                },
            }
        )

    return experiments


def run_experiment_grid() -> None:
    grid = build_experiment_grid()
    total = len(grid)

    for idx, exp in enumerate(grid, start=1):
        model_name: str = exp["model_name"]
        override_params: Dict[str, Any] = exp["override_params"]

        algo = override_params.get("type", model_name)
        run_name = f"{algo}_run_{idx:02d}"

        tags = {
            "algorithm": algo,
            "hw": "4",
            "experiment": "grid",
        }

        print(
            f"\n=== Running experiment {idx}/{total}: {run_name} "
            f"(model={model_name}) ==="
        )

        train_baseline(
            config=TRAINING_CONFIG,
            model_name=model_name,
            override_model_params=override_params,
            run_name=run_name,
            mlflow_tags=tags,
        )


if __name__ == "__main__":
    run_experiment_grid()
