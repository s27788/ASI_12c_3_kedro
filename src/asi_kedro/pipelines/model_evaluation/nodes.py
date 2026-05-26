import logging
import os

import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

load_dotenv()

logger = logging.getLogger(__name__)


def evaluate_model(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Ewaluuje model na zbiorze walidacyjnym.

    Args:
        model: Wytrenowany model RandomForestClassifier.
        X_val: Cechy zbioru walidacyjnego.
        y_val: Target zbioru walidacyjnego.

    Returns:
        Slownik z metrykami: accuracy, f1.
    """
    X_val = X_val.copy()
    category_mappings = getattr(model, "category_mappings_", {})
    for col, mapping in category_mappings.items():
        if col in X_val.columns:
            X_val[col] = X_val[col].astype(str).map(mapping).fillna(-1).astype(int)

    y_pred = model.predict(X_val)

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred)),
    }

    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("F1 Score: %.4f", metrics["f1"])

    try:
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name="baseline-random-forest",
            config={"model_type": "RandomForestClassifier"},
            tags=["baseline", "random_forest"],
        )
        wandb.log(metrics)
    finally:
        wandb.finish()

    return metrics
