from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import TRAINING_CONFIG, TrainingConfig


def predict_from_dataframe(
    model: Pipeline,
    df: pd.DataFrame,
    config: TrainingConfig = TRAINING_CONFIG,
) -> pd.Series:
    """Run predictions on a dataframe that has the same feature columns
    as the training data.

    Parameters
    ----------
    model:
        Trained sklearn Pipeline.
    df:
        Dataframe containing at least the feature columns.
    config:
        Training configuration with feature column names.

    Returns
    -------
    pd.Series
        Predicted labels.
    """
    X = df[config.feature_cols]
    preds = model.predict(X)
    return pd.Series(preds, index=df.index, name="prediction")


def predict_from_records(
    model: Pipeline,
    records: Iterable[dict],
    config: TrainingConfig = TRAINING_CONFIG,
) -> pd.Series:
    """Run predictions on a list of dicts with feature values."""
    df = pd.DataFrame(list(records))
    df = df[config.feature_cols]
    return predict_from_dataframe(model, df, config=config)
