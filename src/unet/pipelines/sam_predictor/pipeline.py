"""
This is a boilerplate pipeline 'sam_predictor'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import initialize_sam, prepare_cropped_images

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=initialize_sam,
            inputs=["params:sam"],
            outputs="sam_predictor",
            name="initialize_sam",
        ),
        node(
            func=prepare_cropped_images,
            inputs=["In_focus_partition", "general_roi_coordinates"],
            outputs="cropped_images_in_focus",
            name="prepare_cropped_images_in_focus",
        ),
        node(
            func=prepare_cropped_images,
            inputs=["Slight_under_focus_partition", "general_roi_coordinates"],
            outputs="cropped_images_slight_under",
            name="prepare_cropped_images_slight_under",
        ),
    ])
