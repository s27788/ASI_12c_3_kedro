import logging
import os
import shutil
from typing import Any, Dict

import pandas as pd
import wandb
from autogluon.tabular import TabularPredictor
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

METRICS_NEGATED = [
    "root_mean_squared_error",
    "mean_absolute_error",
    "mean_squared_error",
    "median_absolute_error",
]

LEADERBOARD_COLUMNS = ["model", "score_val", "pred_time_val", "fit_time"]


def train_automl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: Dict[str, Any],
) -> TabularPredictor:
    """Trenuje modele AutoML za pomoca AutoGluon TabularPredictor.

    AutoGluon samodzielnie zapisuje wszystkie modele na dysku w katalogu
    wskazanym przez parameters['automl']['path'].

    Args:
        X_train: Cechy zbioru treningowego.
        y_train: Wartosci docelowe zbioru treningowego.
        parameters: Slownik parametrow z parameters.yml.

    Returns:
        Wytrenowany TabularPredictor.
    """
    automl_params = parameters["automl"]
    target_column = parameters["target_column"]
    model_path = automl_params["path"]

    if automl_params.get("clean_path", False) and os.path.isdir(model_path):
        shutil.rmtree(model_path)
        logger.info("AutoGluon: usunieto istniejacy katalog modeli: %s", model_path)

    train_data = pd.concat([X_train, y_train], axis=1)

    logger.info(
        "AutoGluon: trening z presetem '%s', time_limit=%ds, path=%s",
        automl_params["presets"],
        automl_params["time_limit"],
        model_path,
    )

    predictor = TabularPredictor(
        label=target_column,
        eval_metric=automl_params["eval_metric"],
        path=model_path,
    ).fit(
        train_data,
        time_limit=automl_params["time_limit"],
        presets=automl_params["presets"],
        verbosity=automl_params["verbosity"],
    )

    logger.info("AutoGluon: trening zakonczony.")
    return predictor


def evaluate_automl(
    predictor: TabularPredictor,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Ewaluuje AutoGluon na zbiorze walidacyjnym i loguje wyniki do W&B.

    Loguje do W&B metryke najlepszego modelu oraz pelny leaderboard (Tables).

    Args:
        predictor: Wytrenowany TabularPredictor.
        X_val: Cechy zbioru walidacyjnego.
        y_val: Wartosci docelowe zbioru walidacyjnego.
        parameters: Slownik parametrow z parameters.yml.

    Returns:
        Slownik z metrykami najlepszego modelu.
    """
    automl_params = parameters["automl"]
    eval_metric = automl_params["eval_metric"]

    val_data = pd.concat([X_val, y_val], axis=1)
    leaderboard = predictor.leaderboard(data=val_data, silent=True)

    best_model_name = leaderboard.iloc[0]["model"]
    best_score_raw = leaderboard.iloc[0]["score_val"]
    best_metric_value = (
        -best_score_raw if eval_metric in METRICS_NEGATED else best_score_raw
    )

    logger.info(
        "AutoGluon: najlepszy model = %s, %s = %.4f",
        best_model_name,
        eval_metric,
        best_metric_value,
    )

    try:
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            name=(
                f"automl-{automl_params['presets']}-"
                f"{automl_params['time_limit']}s"
            ),
            config={
                "model_type": "AutoGluon",
                "presets": automl_params["presets"],
                "time_limit": automl_params["time_limit"],
                "eval_metric": eval_metric,
                "best_model": best_model_name,
            },
            tags=["automl", "autogluon"],
        )

        wandb.log({eval_metric: best_metric_value})

        leaderboard_columns = [
            column
            for column in LEADERBOARD_COLUMNS
            if column in leaderboard.columns
        ]
        leaderboard_table = wandb.Table(
            dataframe=leaderboard[leaderboard_columns].reset_index(drop=True)
        )
        wandb.log({"leaderboard": leaderboard_table})
    finally:
        wandb.finish()

    return {
        "best_model": best_model_name,
        eval_metric: float(best_metric_value),
        "n_models_trained": len(leaderboard),
        "presets": automl_params["presets"],
        "time_limit": automl_params["time_limit"],
    }
