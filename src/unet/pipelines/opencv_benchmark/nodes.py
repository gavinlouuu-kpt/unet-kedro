"""
OpenCV image processing pipeline for contour detection.
"""

import cv2
import numpy as np
from typing import Dict, Any, Callable, Tuple, List
from PIL import Image
import logging
import time
import pandas as pd
from unet.utils.parse_label_json import LabelParser
from unet.utils.dataset import _standardize_key, _validate_image_shapes, filter_empty_frames
logger = logging.getLogger(__name__)


def select_roi(
    partition: Dict[str, Callable[[], Any]]
) -> pd.DataFrame:
    """
    Interactive ROI selection process using OpenCV.
    
    Args:
        partition: Kedro partition containing images as load functions
        
    Returns:
        DataFrame containing ROI coordinates (x, y, width, height)
    """
    # Get the first image for ROI selection
    first_key = next(iter(partition.keys()))
    first_image = partition[first_key]()
    image = np.array(first_image)
    
    # Create window and instructions
    window_name = 'ROI Selection'
    cv2.namedWindow(window_name)
    print("Instructions:")
    print("1. Draw a rectangle by clicking and dragging")
    print("2. Press SPACE or ENTER to confirm selection")
    print("3. Press 'c' to cancel and retry")
    
    # Get ROI using selectROI
    x, y, w, h = cv2.selectROI(window_name, image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    # Create DataFrame with ROI coordinates
    roi_df = pd.DataFrame({
        'x': [x],
        'y': [y],
        'width': [w],
        'height': [h]
    })
    
    logger.info(f"Selected ROI: x={x}, y={y}, width={w}, height={h}")
    return roi_df



def process_image_partition(
    partition: Dict[str, Callable[[], Any]],
    parameters: Dict[str, Any],
    roi: pd.DataFrame
) -> Tuple[Dict[str, Image.Image], pd.DataFrame]:
    """
    Filter empty frames and process all images in a partition within the selected ROI and record processing times.
    
    Args:
        partition: Kedro partition containing images as load functions
        parameters: Parameters containing kernel_size, blur_size, and threshold
        roi: DataFrame containing ROI coordinates (x, y, width, height)
    
    Returns:
        Tuple containing:
        - Dictionary mapping original partition keys to PIL Images
        - DataFrame with processing times in microseconds
    """

    logger.info("Processing images in partition")
    
    # Extract ROI coordinates from DataFrame
    x = roi['x'].iloc[0]
    y = roi['y'].iloc[0]
    w = roi['width'].iloc[0]
    h = roi['height'].iloc[0]
    
    # Setup
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, parameters["kernel_size"])
    processed_images = {}
    timing_data = []
    
    # Get background image
    bg_key = next(key for key in partition.keys() if "background" in key.lower())
    background_pil = partition[bg_key]()
    background = np.array(background_pil)
    
    if background is None:
        raise ValueError(f"Failed to load background image")
    
    # Crop background to ROI
    background = background[y:y+h, x:x+w]
    
    # Process each image
    for key, load_func in partition.items():
        if "background" in key.lower():
            continue
            
        times = {"image_name": key}
        
        # Load and crop image to ROI
        image_pil = load_func()
        image = np.array(image_pil)
        image = image[y:y+h, x:x+w]
        
        # Blur timing
        start_time = time.perf_counter_ns()
        blurred_bg = cv2.GaussianBlur(background, parameters["blur_size"], 0)
        blurred = cv2.GaussianBlur(image, parameters["blur_size"], 0)
        times["blur_time"] = (time.perf_counter_ns() - start_time) / 1000
        
        # Subtraction timing
        start_time = time.perf_counter_ns()
        bg_sub = cv2.subtract(blurred_bg, blurred)
        times["subtraction_time"] = (time.perf_counter_ns() - start_time) / 1000
        
        # Threshold timing
        start_time = time.perf_counter_ns()
        _, binary = cv2.threshold(bg_sub, parameters["threshold"], 255, cv2.THRESH_BINARY)
        times["threshold_time"] = (time.perf_counter_ns() - start_time) / 1000
        
        # Morphology timing
        start_time = time.perf_counter_ns()
        dilate1 = cv2.dilate(binary, kernel, iterations=1)
        erode1 = cv2.erode(dilate1, kernel, iterations=1)
        erode2 = cv2.erode(erode1, kernel, iterations=1)
        processed = cv2.dilate(erode2, kernel, iterations=1)
        times["morphology_time"] = (time.perf_counter_ns() - start_time) / 1000
        
        # Ensure the processed image is properly converted
        processed = processed.astype(np.uint8)
        # logger.info(f"Processed shape before PIL conversion: {processed.shape}")
        processed_images[key] = Image.fromarray(processed, mode='L')  # 'L' mode for grayscale
        
        # Calculate total time
        times["total_time"] = sum(v for k, v in times.items() if k != "image_name")
        
        # Add to timing data
        timing_data.append(times)
    
    timing_df = pd.DataFrame(timing_data)
    
    # Round all numeric columns to 2 decimal places
    numeric_columns = timing_df.select_dtypes(include=['float64']).columns
    timing_df[numeric_columns] = timing_df[numeric_columns].round(2)
    
    return processed_images, timing_df

def load_label_masks(
    labels: Dict[str, Callable[[], Any]]
) -> Dict[str, Image.Image]:
    """
    Load label masks from a dictionary of load functions.
    
    Args:
        labels: Dictionary mapping image keys to load functions that return PIL Images
        
    Returns:
        Dictionary mapping keys to loaded PIL Image masks
        
    Raises:
        ValueError: If masks have inconsistent shapes
    """
    logger.info(f"Loading {len(labels)} label masks")
    
    masks, reference_shape = _validate_image_shapes(
        labels, 
        condition="masks"
    )
    
    logger.info(f"Loaded {len(masks)} masks with shape {reference_shape}")
    return masks


def compare_masks_cv2_sam(
    cv_processed: Dict[str, Callable[[], Any]],
    reconstructed_sam_masks: Dict[str, Image.Image]
) -> pd.DataFrame:
    """
    Compare OpenCV processed masks with reconstructed SAM masks.
    Both inputs are Kedro PartitionedDatasets containing TIFF images.
    """
    logger.info("Comparing OpenCV masks with reconstructed SAM masks")
    
    # Convert images to numpy arrays first, making sure to call the load functions
    cv_arrays = {}
    for k, load_func in cv_processed.items():
        try:
            standardized_key = _standardize_key(k)
            # Call the load function and convert to numpy array
            img = load_func()  # Actually load the image
            if isinstance(img, Image.Image):
                cv_arrays[standardized_key] = np.array(img)
            else:
                cv_arrays[standardized_key] = np.array(img)
            logger.debug(f"Loaded CV mask {k} with shape {cv_arrays[standardized_key].shape}")
        except Exception as e:
            logger.error(f"Error loading CV image {k}: {str(e)}")
            continue
    
    sam_arrays = {}
    for k, load_func in reconstructed_sam_masks.items():
        try:
            standardized_key = _standardize_key(k)
            # Call the load function and convert to numpy array
            img = load_func()  # Actually load the image
            if isinstance(img, Image.Image):
                sam_arrays[standardized_key] = np.array(img)
            else:
                sam_arrays[standardized_key] = np.array(img)
            logger.debug(f"Loaded SAM mask {k} with shape {sam_arrays[standardized_key].shape}")
        except Exception as e:
            logger.error(f"Error loading SAM mask {k}: {str(e)}")
            continue

    # Get common keys
    common_keys = set(cv_arrays.keys()) & set(sam_arrays.keys())
    
    if not common_keys:
        logger.warning("No matching image keys found between CV and SAM datasets")
        return pd.DataFrame()
    
    logger.info(f"Found {len(common_keys)} matching image pairs")
    
    comparison_data = []
    
    for key in sorted(common_keys):
        try:
            cv_mask = cv_arrays[key]
            sam_mask = sam_arrays[key]
            
            # Add shape debugging
            logger.debug(f"Processing pair {key}:")
            logger.debug(f"CV mask shape: {cv_mask.shape}, dtype: {cv_mask.dtype}")
            logger.debug(f"SAM mask shape: {sam_mask.shape}, dtype: {sam_mask.dtype}")
            
            # Ensure arrays are properly loaded
            if cv_mask is None or sam_mask is None:
                logger.warning(f"Skipping {key} - null mask detected")
                continue
            
            # Convert to binary masks (handle both 0-1 and 0-255 ranges)
            cv_mask = (cv_mask > 127).astype(np.uint8)
            sam_mask = (sam_mask > 127).astype(np.uint8)
            
            # Calculate metrics
            intersection = np.logical_and(cv_mask, sam_mask)
            union = np.logical_or(cv_mask, sam_mask)
            
            intersection_sum = np.sum(intersection)
            union_sum = np.sum(union)
            cv_sum = np.sum(cv_mask)
            sam_sum = np.sum(sam_mask)
            
            iou = intersection_sum / union_sum if union_sum > 0 else 0
            dice = (2 * intersection_sum) / (cv_sum + sam_sum) if (cv_sum + sam_sum) > 0 else 0
            
            comparison_data.append({
                'image_name': key,
                'iou_score': round(float(iou), 4),
                'dice_score': round(float(dice), 4),
                'cv_mask_area': int(cv_sum),
                'sam_mask_area': int(sam_sum),
                'intersection_area': int(intersection_sum),
                'union_area': int(union_sum)
            })
            
        except Exception as e:
            logger.error(f"Error processing image pair {key}: {str(e)}")
            logger.exception("Full traceback:")
            continue
    
    if not comparison_data:
        logger.warning("No valid image pairs were processed")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(comparison_data)
    
    # Log summary statistics
    logger.info(f"Successfully processed {len(results_df)} image pairs")
    logger.info(f"Average IoU: {results_df['iou_score'].mean():.4f}")
    logger.info(f"Average Dice: {results_df['dice_score'].mean():.4f}")
    
    return results_df

def reconstruct_sam_masks(
    segmentation_labels_json: List[Dict],
    roi: pd.DataFrame
) -> Dict[str, Image.Image]:
    """
    Convert Label Studio JSON annotations to TIFF images and crop to ROI.
    
    Args:
        segmentation_labels_json: Raw JSON data from Label Studio annotations
        roi: DataFrame containing ROI coordinates (x, y, width, height)
        
    Returns:
        Dictionary mapping image keys to PIL Images ready for saving as TIFF
    """
    # Extract ROI coordinates from DataFrame
    x = roi['x'].iloc[0]
    y = roi['y'].iloc[0]
    w = roi['width'].iloc[0]
    h = roi['height'].iloc[0]
    
    # Use existing parser to decode masks
    masks = LabelParser.parse_json(segmentation_labels_json)
    
    # Convert numpy arrays to PIL Images and crop to ROI
    tiff_masks = {}
    for key, mask in masks.items():
        # Crop mask to ROI
        cropped_mask = mask[y:y+h, x:x+w]
        # Scale binary mask to 0-255 for better visibility
        mask_image = Image.fromarray(cropped_mask * 255)
        tiff_masks[key] = mask_image
    
    logger.info(f"Successfully converted {len(tiff_masks)} masks to TIFF format")
    return tiff_masks