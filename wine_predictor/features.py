from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import TRAINING_CONFIG, TrainingConfig


def create_features_and_target(
    df: pd.DataFrame, config: TrainingConfig = TRAINING_CONFIG
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features X and target y."""
    X = df[config.feature_cols].copy()
    y = df[config.target_col].copy()

    return X, y
