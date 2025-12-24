from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
DEFAULT_EXPERIMENT_NAME = "wine_quality_hw3"


@dataclass
class ReportPaths:
    md_out: Path = Path("docs/experiments/experiments.md")
    fig_dir: Path = Path("docs/experiments/figures")


def _ensure_dirs(paths: ReportPaths) -> None:
    paths.md_out.parent.mkdir(parents=True, exist_ok=True)
    paths.fig_dir.mkdir(parents=True, exist_ok=True)


def _load_runs(experiment_name: str) -> pd.DataFrame:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return pd.DataFrame()

    return mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=200,
    )


def _save_plot_accuracy_by_param(
    runs: pd.DataFrame, out_path: Path, param_col: str
) -> bool:
    if (
        runs.empty
        or "metrics.accuracy" not in runs.columns
        or param_col not in runs.columns
    ):
        return False

    df = runs[[param_col, "metrics.accuracy"]].dropna()
    if df.empty:
        return False

    plt.figure()
    plt.scatter(df[param_col], df["metrics.accuracy"])
    plt.title(f"Accuracy vs {param_col}")
    plt.xlabel(param_col)
    plt.ylabel("accuracy")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def generate_report(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    paths: Optional[ReportPaths] = None,
) -> None:
    paths = paths or ReportPaths()
    _ensure_dirs(paths)

    mlflow.set_tracking_uri(tracking_uri)
    runs = _load_runs(experiment_name)

    lines: list[str] = []
    lines.append("# Experiment comparison\n\n")
    lines.append(f"MLflow experiment: `{experiment_name}`\n\n")

    if runs.empty:
        lines.append("No MLflow runs found.\n\n")
        lines.append("Run experiments first:\n\n")
        lines.append("```bash\npoetry run python -m wine_predictor.experiments\n```\n")
        paths.md_out.write_text("".join(lines), encoding="utf-8")
        return

    preferred = [
        "run_id",
        "metrics.accuracy",
        "params.model",
        "params.random_state",
        "params.C",
        "params.n_estimators",
        "params.max_depth",
    ]
    preferred = [c for c in preferred if c in runs.columns]
    view_small = runs[preferred].head(30)

    lines.append("## Top runs\n\n")
    lines.append(view_small.to_markdown(index=False))
    lines.append("\n\n")

    lines.append("## Plots\n\n")

    plot1 = paths.fig_dir / "experiments_accuracy_vs_C.png"
    if _save_plot_accuracy_by_param(runs, plot1, "params.C"):
        lines.append(f"![Accuracy vs C](../figures/{plot1.name})\n\n")

    plot2 = paths.fig_dir / "experiments_accuracy_vs_max_depth.png"
    if _save_plot_accuracy_by_param(runs, plot2, "params.max_depth"):
        lines.append(f"![Accuracy vs max_depth](../figures/{plot2.name})\n\n")

    plot3 = paths.fig_dir / "experiments_accuracy_vs_n_estimators.png"
    if _save_plot_accuracy_by_param(runs, plot3, "params.n_estimators"):
        lines.append(f"![Accuracy vs n_estimators](../figures/{plot3.name})\n\n")

    paths.md_out.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    generate_report()
