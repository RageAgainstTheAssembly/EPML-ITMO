from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import OmegaConf

SUPPORTED_MODEL_TYPES = {"logreg", "random_forest", "gradient_boosting"}


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

    if not 0.0 < test_size < 1.0:
        raise ValueError(
            f"Invalid train.test_size={test_size!r}. "
            "Expected a float in the (0, 1) range."
        )

    if random_state < 0:
        raise ValueError(
            f"Invalid train.random_state={random_state!r}. "
            "Expected a non-negative integer."
        )

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
    container = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[no-any-return]

    if "model" not in container or not isinstance(container["model"], dict):
        raise ValueError(f"Config {cfg_path} must contain a top-level 'model' mapping.")

    model_cfg: Dict[str, Any] = container["model"]  # type: ignore[assignment]

    model_type = model_cfg.get("type")
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Invalid or missing model.type={model_type!r} in {cfg_path}. "
            f"Expected one of: {sorted(SUPPORTED_MODEL_TYPES)}."
        )

    if model_type == "logreg":
        C = float(model_cfg.get("C", 1.0))
        if C <= 0:
            raise ValueError(
                f"Invalid C={C!r} for logreg in {cfg_path}. Expected C > 0."
            )

    if model_type == "random_forest":
        n_estimators = int(model_cfg.get("n_estimators", 100))
        if n_estimators <= 0:
            raise ValueError(
                f"Invalid n_estimators={n_estimators!r} for random_forest in {cfg_path}. "
                "Expected n_estimators > 0."
            )

    if model_type == "gradient_boosting":
        gb_lr = float(model_cfg.get("gb_learning_rate", 0.1))
        if gb_lr <= 0:
            raise ValueError(
                f"Invalid gb_learning_rate={gb_lr!r} for gradient_boosting in {cfg_path}. "
                "Expected gb_learning_rate > 0."
            )

    return {"model": model_cfg}


TRAINING_CONFIG = load_training_config()
