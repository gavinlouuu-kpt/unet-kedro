"""
This is a boilerplate pipeline 'sam_predictor'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import initialize_sam, parse_label_studio_json

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=initialize_sam,
            inputs=["params:sam"],
            outputs="sam_predictor",
            name="initialize_sam",
        ),
        node(
            func=parse_label_studio_json,
            inputs="segmentation_labels_in_focus",
            outputs="prompt_points_in_focus",
            name="parse_label_studio_json_in_focus",
        ),
    ])
