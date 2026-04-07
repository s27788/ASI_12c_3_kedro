import logging
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


def evaluate_model(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    y_pred = model.predict(X_val)

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred)),
    }

    logger.info("Accuracy: %.4f", metrics["accuracy"])
    logger.info("F1 Score: %.4f", metrics["f1"])

    return metrics