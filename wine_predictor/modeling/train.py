from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import TRAINING_CONFIG, TrainingConfig
from ..dataset import load_wine_data
from ..features import create_features_and_target


def build_logreg_pipeline() -> Pipeline:
    """Create a baseline logistic regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    n_jobs=-1,
                    multi_class="auto",
                ),
            ),
        ]
    )


def train_baseline(
    *,
    config: TrainingConfig = TRAINING_CONFIG,
) -> Pipeline:
    """Train a baseline logistic regression model on WineQT.csv.

    Returns
    -------
    Pipeline
        Trained sklearn Pipeline (StandardScaler + LogisticRegression).
    """
    df = load_wine_data(config=config)
    X, y = create_features_and_target(df, config=config)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    model = build_logreg_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model


if __name__ == "__main__":
    train_baseline()
