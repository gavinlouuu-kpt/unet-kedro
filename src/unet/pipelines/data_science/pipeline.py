"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import test_node, split_data

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["segmentation_dataset_in_focus", "params:split_ratio"],
            outputs=["train_dataset", "val_dataset"],
            name="split_data_node"
        )
    ])
