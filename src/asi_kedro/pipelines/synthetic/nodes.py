import logging

import pandas as pd
import wandb
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

logger = logging.getLogger(__name__)


def generate_synthetic_data(real_data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Generuje dane syntetyczne synthesizerem GaussianCopula.

    Args:
        real_data: Oryginalny zbiór danych.
        parameters: Sekcja params:synthetic_config z parameters.yml.

    Returns:
        Wygenerowany zbiór syntetyczny.
    """
    metadata = Metadata.detect_from_dataframe(data=real_data)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)

    num_rows = parameters.get("synthetic_rows", 100)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    logger.info("Wygenerowano %d rekordów syntetycznych", len(synthetic_data))
    return synthetic_data


def evaluate_synthetic_data(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    parameters: dict,
) -> dict:
    """Ewaluuje jakość danych syntetycznych i loguje wyniki do W&B.

    Args:
        real_data: Dane rzeczywiste.
        synthetic_data: Dane wygenerowane przez SDV.
        parameters: Sekcja params:synthetic_config.

    Returns:
        Słownik z diagnostic_score i quality_score.
    """
    metadata = Metadata.detect_from_dataframe(data=real_data)
    real_clean = real_data.dropna()
    synth_clean = synthetic_data.dropna()

    diagnostic = run_diagnostic(
        real_data=real_clean,
        synthetic_data=synth_clean,
        metadata=metadata,
    )
    quality = evaluate_quality(
        real_data=real_clean,
        synthetic_data=synth_clean,
        metadata=metadata,
    )

    scores = {
        "diagnostic_score": float(diagnostic.get_score()),
        "quality_score": float(quality.get_score()),
    }

    with wandb.init(
        project=parameters.get("wandb_project", "asi-pjatk"),
        entity=parameters.get("wandb_entity"),
        job_type="sdv_evaluation",
        config={"n_samples": len(synthetic_data)},
    ):
        wandb.log({
            "sdv/diagnostic_score": scores["diagnostic_score"],
            "sdv/quality_score": scores["quality_score"],
        })

    return scores
