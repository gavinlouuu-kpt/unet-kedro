"""
This is a boilerplate pipeline 'opencv_benchmark'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import process_image_partition, compare_masks_cv2_sam, reconstruct_sam_masks

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=process_image_partition,
                inputs=["In_focus_partition", "params:cv_basic_options"],
                outputs=["cv_processed_in_focus", "cv_processed_timing_in_focus"],
                name="cv_process_in_focus_images",
            ),
            node(
                func=process_image_partition,
                inputs=["Slight_under_focus_partition", "params:cv_basic_options"],
                outputs=["cv_processed_slight_under", "cv_processed_timing_slight_under"],
                name="cv_process_slight_under_images",
            ),
            node(
                func=reconstruct_sam_masks,
                inputs="segmentation_labels_in_focus",
                outputs="reconstructed_sam_masks_in_focus",
                name="reconstruct_sam_masks_in_focus",
            ),
            node(
                func=reconstruct_sam_masks,
                inputs="segmentation_labels_slight_under",
                outputs="reconstructed_sam_masks_slight_under",
                name="reconstruct_sam_masks_slight_under",
            ),
            node(
                func=compare_masks_cv2_sam,
                inputs=["cv_processed_in_focus", "segmentation_labels_in_focus"],
                outputs="mask_comparison_metrics",
                name="compare_masks"
            ),
            node(
                func=compare_masks_cv2_sam,
                inputs=["cv_processed_slight_under", "segmentation_labels_slight_under"],
                outputs="mask_comparison_metrics_slight_under",
                name="compare_masks_slight_under"
            )
        ]
    )