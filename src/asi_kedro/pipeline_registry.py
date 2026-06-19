"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()

    baseline = (
        pipelines["data_processing"]
        + pipelines["model_training"]
        + pipelines["model_evaluation"]
    )

    return {
        **pipelines,
        "baseline": baseline,
        "__default__": baseline,
        "automl": pipelines["automl"],
        "full": baseline + pipelines["automl"],
        "synthetic": pipelines["synthetic"]
    }
