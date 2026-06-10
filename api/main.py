import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.model_loader import (
    AUTOGLUON_MODEL_RELATIVE_PATH,
    PROJECT_ROOT,
    get_predictor,
    is_model_loaded,
    load_model,
    unload_model,
)
from api.schemas import ModelInfoResponse, PassengerFeatures, PredictionResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    unload_model()


app = FastAPI(
    title="Airline Satisfaction API",
    description="API for serving ML predictions (Sprint 5).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, bool | str]:
    """Return API status and whether the prediction model is loaded."""
    return {
        "status": "ok",
        "model_loaded": is_model_loaded(),
    }


def _extract_prediction_value(prediction: object) -> int:
    if hasattr(prediction, "iloc"):
        value = prediction.iloc[0]
    else:
        value = prediction[0]
    return int(value)


def _get_model_name(predictor: object) -> str:
    model_name = getattr(predictor, "model_best", None)
    return model_name if model_name else "autogluon"


def _get_predictor_features(predictor: object) -> list[str]:
    feature_metadata = getattr(predictor, "feature_metadata", None)
    if feature_metadata is None:
        return []

    get_features = getattr(feature_metadata, "get_features", None)
    if get_features is None:
        return []

    try:
        return list(get_features())
    except Exception:
        logger.exception("Failed to read feature names from loaded predictor.")
        return []


def _format_model_path(saved_path: object) -> str:
    if saved_path is None:
        return AUTOGLUON_MODEL_RELATIVE_PATH

    path = Path(str(saved_path))
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _build_model_info_response() -> ModelInfoResponse:
    predictor = get_predictor()
    if predictor is None:
        return ModelInfoResponse(loaded=False)

    features = _get_predictor_features(predictor)
    problem_type = getattr(predictor, "problem_type", None)
    label = getattr(predictor, "label", None)
    eval_metric = getattr(predictor, "eval_metric", None)
    saved_path = getattr(predictor, "path", None)

    return ModelInfoResponse(
        loaded=True,
        model_name=_get_model_name(predictor),
        model_type=type(predictor).__name__,
        problem_type=str(problem_type) if problem_type is not None else None,
        label=str(label) if label is not None else None,
        eval_metric=str(eval_metric) if eval_metric is not None else None,
        model_path=_format_model_path(saved_path),
        feature_count=len(features) if features else None,
        features=features or None,
    )


@app.get("/model-info", response_model=ModelInfoResponse, response_model_exclude_none=True)
def model_info() -> ModelInfoResponse:
    """Return metadata about the loaded prediction model."""
    return _build_model_info_response()


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PassengerFeatures) -> PredictionResponse:
    """Predict passenger satisfaction (0 or 1) from input features."""
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded.",
        )

    try:
        input_df = pd.DataFrame([features.model_dump(by_alias=True)])
        prediction = predictor.predict(input_df)
        return PredictionResponse(
            prediction=_extract_prediction_value(prediction),
            model_name=_get_model_name(predictor),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Prediction failed.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from None
