from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate_and_log


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=evaluate_and_log,
            inputs=["classifier", "X_val", "y_val", "parameters"],
            outputs="metrics",
            name="evaluate_and_log_node",
        )
    ])
