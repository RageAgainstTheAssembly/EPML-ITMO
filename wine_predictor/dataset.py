from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TRAINING_CONFIG, TrainingConfig


def load_wine_data(config: TrainingConfig = TRAINING_CONFIG) -> pd.DataFrame:
    """Load WineQT dataset from disk and perform basic validation."""
    csv_path: Path = config.paths.wine_csv

    if not csv_path.exists():
        raise FileNotFoundError(
            f"WineQT.csv not found at {csv_path}. "
            "Make sure the file is placed in data/external/WineQT.csv"
        )

    df = pd.read_csv(csv_path)

    required_cols = set(config.feature_cols + [config.target_col])
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing expected columns in WineQT.csv: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )

    return df
