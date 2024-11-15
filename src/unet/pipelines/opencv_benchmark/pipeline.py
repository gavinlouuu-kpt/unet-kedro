"""
This is a boilerplate pipeline 'opencv_benchmark'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline
from .nodes import (process_image, compare_masks_cv2_sam, 
                   reconstruct_sam_masks, select_roi, create_mask_overlays)
from unet.utils.dataset import filter_empty_frames

def create_pipeline(**kwargs) -> Pipeline:
    # Define the base pipeline that will be reused
    base_pipeline = pipeline(
        [
            node(
                func=filter_empty_frames,
                inputs=["original_partition", "params:cv_basic_options", "general_roi_coordinates"],
                outputs="filtered_images",
                name="filter_empty_frames",
            ),
            node(
                func=process_image,
                inputs=["filtered_images", "params:cv_basic_options", "general_roi_coordinates"],
                outputs=["cv_processed", "cv_processed_timing"],
                name="cv_process_images",
            ),
            node(
                func=reconstruct_sam_masks,
                inputs=["segmentation_labels", "general_roi_coordinates"],
                outputs="reconstructed_sam_masks",
                name="reconstruct_sam_masks",
            ),
            node(
                func=compare_masks_cv2_sam,
                inputs=["cv_processed", "reconstructed_sam_masks"],
                outputs="mask_comparison_metrics",
                name="compare_masks",
            ),
            node(
                func=create_mask_overlays,
                inputs=[
                    "cv_processed",
                    "reconstructed_sam_masks",
                    "filtered_images",
                    "general_roi_coordinates"
                ],
                outputs=[
                    "cv_overlays",
                    "sam_overlays",
                    "combined_overlays"
                ],
                name="create_overlays"
            )
        ]
    )

    # Common nodes should run before the namespaced pipelines
    common_nodes = pipeline(
        [
            node(
                func=select_roi,
                inputs="original_partition",  # This will use In_focus data without namespace
                outputs="general_roi_coordinates",
                name="select_roi",
            ),
        ],
        namespace="In_focus"  # Add namespace to ensure it uses In_focus data
    )

    # Create pipeline instances with different namespaces
    in_focus_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="In_focus",
    )

    slight_under_pipeline = pipeline(
        pipe=base_pipeline,
        namespace="Slight_under_focus",
    )

    return in_focus_pipeline + slight_under_pipeline # no roi selection
    # return common_nodes + in_focus_pipeline + slight_under_pipeline