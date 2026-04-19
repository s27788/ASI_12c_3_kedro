import logging
import os

import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

load_dotenv()

logger = logging.getLogger(__name__)


def evaluate_and_log(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    parameters: dict,
) -> dict:
    """Evaluate the model on validation data and log metrics to Weights & Biases.

    Args:
        model: Trained RandomForestClassifier.
        X_val: Validation features.
        y_val: Validation target values.
        parameters: Parameters loaded from Kedro parameters.yml.

    Returns:
        Dictionary with evaluation metrics.
    """
    wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        config=parameters,
        name=f"rf-{parameters['model']['n_estimators']}",
    )

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

    wandb.log(metrics)
    wandb.finish()

    return metrics
