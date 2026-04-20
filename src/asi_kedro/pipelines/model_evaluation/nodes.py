import logging
import os

import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_PATH = "data/06_models/random_forest.pkl"


def evaluate_and_log(
    model: RandomForestClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    parameters: dict,
) -> dict:
    """Evaluate the model on validation data and log metrics to Weights & Biases.

    Oprocz metryk loguje rowniez wytrenowany model jako wandb.Artifact,
    dzieki czemu w zakladce Artifacts pojawia sie kolejna wersja (v0, v1, ...).

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
    try:
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

        # --- Logowanie artefaktu modelu do W&B ---
        # Model zapisuje sie na dysk przez Kedro (PickleDataset w catalog.yml),
        # wiec tutaj dolaczamy ten plik .pkl do runu jako artefakt typu "model".
        model_params = parameters["model"]
        if os.path.exists(MODEL_PATH):
            artifact = wandb.Artifact(
                name="airline-satisfaction-model",
                type="model",
                description=(
                    f"RandomForest (n_estimators={model_params['n_estimators']}, "
                    f"max_depth={model_params['max_depth']})"
                ),
                metadata={
                    "n_estimators": model_params["n_estimators"],
                    "max_depth": model_params["max_depth"],
                    "min_samples_split": model_params["min_samples_split"],
                    "random_state": model_params["random_state"],
                    "accuracy": metrics["accuracy"],
                    "f1": metrics["f1"],
                },
            )
            artifact.add_file(MODEL_PATH)
            wandb.log_artifact(artifact)
            logger.info("Zalogowano artefakt modelu z pliku %s", MODEL_PATH)
        else:
            logger.warning(
                "Nie znaleziono pliku modelu pod %s - artefakt nie zostanie zalogowany.",
                MODEL_PATH,
            )

        return metrics
    finally:
        wandb.finish()