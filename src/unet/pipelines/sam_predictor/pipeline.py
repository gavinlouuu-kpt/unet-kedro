"""
This is a boilerplate pipeline 'sam_predictor'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import initialize_sam, prepare_cropped_images, predict_masks

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
            inputs=["filtered_in_focus_partition", "general_roi_coordinates"],
            outputs="cropped_images_in_focus",
            name="prepare_cropped_images_in_focus",
        ),
        node(
            func=prepare_cropped_images,
            inputs=["filtered_slight_under_partition", "general_roi_coordinates"],
            outputs="cropped_images_slight_under",
            name="prepare_cropped_images_slight_under",
        ),
        node(
            func=predict_masks,
            inputs=["sam_predictor", "cropped_images_in_focus"],
            outputs={"masks": "prompted_sam_masks_in_focus",
                    "timing": "sam_prediction_timing_in_focus"},
            name="predict_masks_in_focus",
        ),
        node(
            func=predict_masks,
            inputs=["sam_predictor", "cropped_images_slight_under"],
            outputs={"masks": "prompted_sam_masks_slight_under",
                    "timing": "sam_prediction_timing_slight_under"},
            name="predict_masks_slight_under",
        ),
    ])
