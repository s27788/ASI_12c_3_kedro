from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_data, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["raw_data", "parameters"],
                outputs="processed_data",
                name="preprocess_node",
            ),
            
            node(
                            func=split_data,
                            inputs=["processed_data", "parameters"],
                            outputs=[
                                "X_train", "X_val", "X_test",
                                "y_train", "y_val", "y_test",
                            ],
                            name="split_data_node",
                        ),
        ]
    )