"""
This is a boilerplate pipeline 'opencv_benchmark'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, pipeline, node
from .nodes import (process_image, compare_masks_cv2_sam, 
                    reconstruct_sam_masks, select_roi, create_mask_overlays)
from unet.utils.dataset import filter_empty_frames

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=select_roi,
                inputs="original_In_focus_partition",
                outputs="general_roi_coordinates",
                name="select_roi",
            ),
            node(
                func=filter_empty_frames,
                inputs=["original_In_focus_partition", "params:cv_basic_options", "general_roi_coordinates"],
                outputs="filtered_In_focus",
                name="filter_empty_frames_In_focus",
            ),
            node(
                func=process_image,
                inputs=["filtered_In_focus", "params:cv_basic_options", "general_roi_coordinates"],
                outputs=["cv_processed_In_focus", "cv_processed_timing_In_focus"],
                name="cv_process_In_focus_images",
            ),
            node(
                func=filter_empty_frames,
                inputs=["original_Slight_under_focus_partition", "params:cv_basic_options", "general_roi_coordinates"],
                outputs="filtered_Slight_under_focus", 
                name="filter_empty_frames_Slight_under_focus",
            ),
            node(
                func=process_image,
                inputs=["filtered_Slight_under_focus", "params:cv_basic_options", "general_roi_coordinates"],
                outputs=["cv_processed_Slight_under_focus", "cv_processed_timing_Slight_under_focus"],
                name="cv_process_Slight_under_focus_images",
            ),
            node(
                func=reconstruct_sam_masks,
                inputs=["segmentation_labels_In_focus", "general_roi_coordinates"],
                outputs="reconstructed_sam_masks_In_focus", 
                name="reconstruct_sam_masks_In_focus",
            ),
            node(
                func=reconstruct_sam_masks,
                inputs=["segmentation_labels_Slight_under_focus", "general_roi_coordinates"],
                outputs="reconstructed_sam_masks_Slight_under_focus",
                name="reconstruct_sam_masks_Slight_under_focus",
            ),
            node(
                func=compare_masks_cv2_sam,
                inputs=["cv_processed_In_focus", "reconstructed_sam_masks_In_focus"],
                outputs="mask_comparison_metrics",
                name="compare_masks"
            ),
            node(
                func=compare_masks_cv2_sam,
                inputs=["cv_processed_Slight_under_focus", "reconstructed_sam_masks_Slight_under_focus"],
                outputs="mask_comparison_metrics_Slight_under_focus",
                name="compare_masks_Slight_under_focus"
            ),
            node(
                func=create_mask_overlays,
                inputs=["cv_processed_In_focus", "reconstructed_sam_masks_In_focus", "filtered_In_focus", "general_roi_coordinates"],
                outputs=["In_focus_cv_overlays", "In_focus_sam_overlays", "In_focus_combined_overlays"],
                name="create_In_focus_overlays"
            ),
            node(
                func=create_mask_overlays,
                inputs=["cv_processed_Slight_under_focus", "reconstructed_sam_masks_Slight_under_focus", "filtered_Slight_under_focus", "general_roi_coordinates"],
                outputs=["Slight_under_focus_cv_overlays", "Slight_under_focus_sam_overlays", "Slight_under_focus_combined_overlays"],
                name="create_Slight_under_focus_overlays"
            )
        ]
    )