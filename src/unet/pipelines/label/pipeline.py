"""
This is a boilerplate pipeline 'label'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_segmentation_dataset

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_segmentation_dataset,
            inputs=["In_focus_partition", "segmentation_labels"],
            outputs="segmentation_dataset",
            name="create_segmentation_dataset_node"
        )
    ])
