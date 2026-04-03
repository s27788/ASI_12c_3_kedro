from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["raw_data", "parameters"],
                outputs="processed_data",
                name="preprocess_node",
            ),
        ]
    )