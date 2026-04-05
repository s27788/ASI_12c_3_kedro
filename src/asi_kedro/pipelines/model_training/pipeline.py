from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model


def create_pipeline(**kwargs) -> Pipeline:
    """Tworzy pipeline trenowania modelu."""
    return pipeline([
        node(
            func=train_model,
            inputs=["X_train", "y_train", "X_val", "y_val", "parameters"],
            outputs="classifier",
            name="train_model_node",
        ),
    ])