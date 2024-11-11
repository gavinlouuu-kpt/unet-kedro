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

logger = logging.getLogger(__name__)

def process_image_partition(
    partition: Dict[str, Callable[[], Any]],
    parameters: Dict[str, Any]
) -> Tuple[Dict[str, Image.Image], pd.DataFrame]:
    """
    Process all images in a partition and record processing times in microseconds.
    
    Args:
        partition: Kedro partition containing images as load functions
        parameters: Parameters containing kernel_size, blur_size, and threshold
    
    Returns:
        Tuple containing:
        - Dictionary mapping original partition keys to PIL Images
        - DataFrame with processing times in microseconds for blur, subtraction, threshold, and morphology operations
    """
    logger.info("Processing images in partition")
    
    # Setup
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, parameters["kernel_size"])
    processed_images = {}
    
    # Setup timing DataFrame
    timing_data = []
    
    # Get background image
    bg_key = next(key for key in partition.keys() if "background" in key.lower())
    background_pil = partition[bg_key]()
    background = np.array(background_pil)
    
    if background is None:
        raise ValueError(f"Failed to load background image")
    
    # Process each image
    for key, load_func in partition.items():
        if "background" in key.lower():
            continue
            
        times = {"image_name": key}
        
        # Load image (not timed)
        image_pil = load_func()
        image = np.array(image_pil)
        
        if image is None:
            logger.error(f"Failed to load image")
            continue
        
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

def _validate_image_shapes(
    images: Dict[str, Callable[[], Any]], 
    condition: str,
    sample_size: int = 5
) -> Tuple[Dict[str, np.ndarray], tuple]:
    """
    Helper function to load images and validate their shapes.
    
    Args:
        images: Dictionary of image load functions
        condition: Name of the condition for logging (e.g., 'in_focus')
        sample_size: Number of images to check for shape consistency
        
    Returns:
        Tuple containing:
        - Dictionary of loaded images as numpy arrays
        - Reference shape of the images
        
    Raises:
        ValueError: If inconsistent shapes are detected
    """
    loaded_images = {}
    reference_shape = None
    
    for i, (key, load_func) in enumerate(images.items()):
        img = np.array(load_func())
        loaded_images[key] = img
        
        # Shape validation for first n images
        if i < sample_size:
            current_shape = img.shape
            if reference_shape is None:
                reference_shape = current_shape
                logger.info(f"{condition} reference shape: {current_shape}")
            elif current_shape != reference_shape:
                raise ValueError(
                    f"Inconsistent shape in {condition} images. "
                    f"Expected {reference_shape}, "
                    f"got {current_shape} for image {key}"
                )
    
    return loaded_images, reference_shape
    
def _standardize_key(key: str) -> str:
    """Standardize key to XXXX format"""
    # Remove 'image.' prefix if present
    if key.startswith('image.'):
        key = key.replace('image.', '')
    # Ensure 4-digit padding
    return key.zfill(4)

def compare_masks_cv2_sam(
    cv_processed: Dict[str, Callable[[], Any]],
    segmentation_labels_json: List[Dict]
) -> pd.DataFrame:
    """
    Compare OpenCV processed masks with SAM segmentation labels from Label Studio JSON.
    """
    logger.info("Comparing OpenCV masks with SAM labels")
    logger.info(f"Total CV processed images: {len(cv_processed)}")
    
    # Parse the Label Studio JSON into masks
    sam_masks = LabelParser.parse_json(segmentation_labels_json)
    
    # Standardize CV image keys and load images
    cv_images = {}
    for key, load_func in cv_processed.items():
        standardized_key = _standardize_key(key)
        cv_images[standardized_key] = load_func()
    
    # Standardize SAM mask keys
    sam_masks = {_standardize_key(k): v for k, v in sam_masks.items()}
    
    # Debug logging
    logger.info(f"CV processed keys: {list(cv_images.keys())[:5]}...")
    logger.info(f"SAM mask keys: {list(sam_masks.keys())[:5]}...")
    logger.info(f"Total CV images after standardization: {len(cv_images)}")
    logger.info(f"Total SAM masks after standardization: {len(sam_masks)}")
    
    # Get common keys
    common_keys = set(cv_images.keys()) & set(sam_masks.keys())
    
    if not common_keys:
        logger.warning("No matching image keys found between CV and SAM datasets")
        logger.info(f"CV image format example: {next(iter(cv_images.keys()), 'no keys')}")
        logger.info(f"SAM mask format example: {next(iter(sam_masks.keys()), 'no keys')}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(common_keys)} matching image pairs")
    
    comparison_data = []
    
    for key in common_keys:
        try:
            # Get corresponding images
            cv_mask = cv_images[key]
            sam_mask = sam_masks[key]
            
            # Convert to numpy arrays if needed
            if isinstance(cv_mask, Image.Image):
                cv_mask = np.array(cv_mask)
            
            # Ensure both masks are binary
            cv_mask = (cv_mask > 0).astype(np.uint8)
            sam_mask = (sam_mask > 0).astype(np.uint8)
            
            # Calculate metrics
            intersection = np.logical_and(cv_mask, sam_mask)
            union = np.logical_or(cv_mask, sam_mask)
            
            iou = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
            dice = (2 * np.sum(intersection)) / (np.sum(cv_mask) + np.sum(sam_mask)) if (np.sum(cv_mask) + np.sum(sam_mask)) > 0 else 0
            
            comparison_data.append({
                'image_name': key,
                'iou_score': round(float(iou), 4),
                'dice_score': round(float(dice), 4)
            })
            
        except Exception as e:
            logger.error(f"Error processing image pair {key}: {str(e)}")
            continue
    
    if not comparison_data:
        logger.warning("No valid image pairs were processed")
        return pd.DataFrame()
    
    df = pd.DataFrame(comparison_data)
    
    # Log summary statistics
    logger.info(f"Successfully processed {len(df)} image pairs")
    logger.info(f"Average IoU: {df['iou_score'].mean():.4f}")
    logger.info(f"Average Dice: {df['dice_score'].mean():.4f}")
    
    return df

def reconstruct_sam_masks(
    segmentation_labels_json: List[Dict]
) -> Dict[str, Image.Image]:
    """
    Convert Label Studio JSON annotations to TIFF images.
    
    Args:
        segmentation_labels_json: Raw JSON data from Label Studio annotations
        
    Returns:
        Dictionary mapping image keys to PIL Images ready for saving as TIFF
    """
    # Use existing parser to decode masks
    masks = LabelParser.parse_json(segmentation_labels_json)
    
    # Convert numpy arrays to PIL Images
    tiff_masks = {}
    for key, mask in masks.items():
        # Scale binary mask to 0-255 for better visibility
        mask_image = Image.fromarray(mask * 255)
        tiff_masks[key] = mask_image
    
    logger.info(f"Successfully converted {len(tiff_masks)} masks to TIFF format")
    return tiff_masks