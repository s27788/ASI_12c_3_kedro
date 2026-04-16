import logging
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import os
import wandb
from dotenv import load_dotenv



load_dotenv()

logger = logging.getLogger(__name__)


def evaluate_and_log(
        model: RandomForestClassifier,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        parameters: dict,
) -> dict:
    """Ewaluuje model i loguje wyniki do W&B."""

    run = wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        entity=os.getenv("WANDB_ENTITY"),
        config=parameters,
        name=f"rf-{parameters['model']['n_estimators']}"
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

    # W&B logowanie
    wandb.log(metrics)

    # artifact (model)
    #artifact = wandb.Artifact(
     #   name="model",
    #    type="model"
    #)

    artifact.add_file("data/06_models/baseline_rf.pkl")
    wandb.log_artifact(artifact)

    wandb.finish()

    return metrics