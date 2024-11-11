"""
This is a boilerplate pipeline 'opencv_benchmark'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import process_image_partition

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=process_image_partition,
                inputs=["In_focus_partition", "params:cv_basic_options"],
                outputs="cv_processed_in_focus",
                name="cv_process_in_focus_images",
            ),
            node(
                func=process_image_partition,
                inputs=["Slight_under_focus_partition", "params:cv_basic_options"],
                outputs="cv_processed_slight_under",
                name="cv_process_slight_under_images",
            ),
        ]
    )