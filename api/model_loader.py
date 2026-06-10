import logging
from pathlib import Path

from autogluon.tabular import TabularPredictor

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUTOGLUON_MODEL_PATH = PROJECT_ROOT / "data" / "06_models" / "autogluon"
AUTOGLUON_MODEL_RELATIVE_PATH = "data/06_models/autogluon"

_predictor: TabularPredictor | None = None


def get_predictor() -> TabularPredictor | None:
    return _predictor


def is_model_loaded() -> bool:
    return _predictor is not None


def load_model() -> None:
    """Load the AutoGluon model once at application startup.

    If the model directory is missing or loading fails, the predictor stays
    None and the error is logged without blocking API startup.
    """
    global _predictor

    if not AUTOGLUON_MODEL_PATH.is_dir():
        logger.warning(
            "AutoGluon model directory not found: %s. "
            "Run Kedro pipeline to train and save the model.",
            AUTOGLUON_MODEL_PATH,
        )
        return

    try:
        _predictor = TabularPredictor.load(str(AUTOGLUON_MODEL_PATH))
        logger.info(
            "AutoGluon model loaded from %s (best model: %s).",
            AUTOGLUON_MODEL_PATH,
            _predictor.model_best,
        )
    except Exception:
        _predictor = None
        logger.exception(
            "Failed to load AutoGluon model from %s.",
            AUTOGLUON_MODEL_PATH,
        )


def unload_model() -> None:
    global _predictor
    _predictor = None
