from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf


@dataclass(frozen=True)
class ProjectPaths:
    """Common project paths resolved from this file location."""

    root: Path = Path(__file__).resolve().parents[1]
    data_external: Path = root / "data" / "external"
    models_dir: Path = root / "models"

    @property
    def wine_csv(self) -> Path:
        return self.data_external / "WineQT.csv"


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
CONFIG_ROOT = PATHS.root / "configs"


def load_training_config() -> TrainingConfig:
    """
    Load training config from configs/train/base.yaml (if present)
    """
    base = TrainingConfig()

    cfg_path = CONFIG_ROOT / "train" / "base.yaml"
    if not cfg_path.exists():
        return base

    cfg = OmegaConf.load(cfg_path)
    train_cfg = cfg.get("train", {})

    test_size = float(train_cfg.get("test_size", base.test_size))
    random_state = int(train_cfg.get("random_state", base.random_state))

    return TrainingConfig(
        paths=base.paths,
        feature_cols=base.feature_cols,
        target_col=base.target_col,
        test_size=test_size,
        random_state=random_state,
    )


def load_model_config(model_name: str) -> Dict[str, Any]:
    """
    Load model-specific configuration from configs/model/<model_name>.yaml.
    """
    cfg_path = CONFIG_ROOT / "model" / f"{model_name}.yaml"
    if not cfg_path.exists():
        raise ValueError(f"Model config not found for: {model_name} at {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[no-any-return]


TRAINING_CONFIG = load_training_config()
