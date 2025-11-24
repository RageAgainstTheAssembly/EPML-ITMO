from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class ProjectPaths:
    """Common project paths resolved from this file location."""

    root: Path = Path(__file__).resolve().parents[1]
    data_raw: Path = root / "data" / "raw"
    models_dir: Path = root / "models"

    @property
    def wine_csv(self) -> Path:
        return self.data_raw / "WineQT.csv"


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for baseline wine quality model training."""

    paths: ProjectPaths = field(default_factory=ProjectPaths)

    feature_cols: List[str] = field(
        default_factory=lambda: [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol",
        ]
    )
    target_col: str = "quality"

    test_size: float = 0.2
    random_state: int = 42


PATHS = ProjectPaths()
TRAINING_CONFIG = TrainingConfig()
