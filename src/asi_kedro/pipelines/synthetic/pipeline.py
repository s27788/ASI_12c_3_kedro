from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_synthetic_data, evaluate_synthetic_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_synthetic_data,
            inputs=["processed_data", "params:synthetic_config"],
            outputs="synthetic_data_csv",
            name="generate_synthetic_data_node",
        ),
        node(
            func=evaluate_synthetic_data,
            inputs=["processed_data", "synthetic_data_csv", "params:synthetic_config"],
            outputs="evaluation_scores",
            name="evaluate_synthetic_data_node",
        )
    ])