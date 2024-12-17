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
from unet.utils.dataset import _standardize_key, _validate_image_shapes, filter_empty_frames, select_roi
logger = logging.getLogger(__name__)



def process_image(
    images: Dict[str, Image.Image],
    parameters: Dict[str, Any],
    roi: pd.DataFrame
) -> Tuple[Dict[str, Image.Image], pd.DataFrame]:
    """
    Process images using OpenCV operations within the selected ROI.
    
    Args:
        images: Dictionary containing images as PIL Images
        parameters: Parameters containing kernel_size, blur_size, and threshold
        roi: DataFrame containing ROI coordinates
    
    Returns:
        Tuple containing processed images dictionary and timing DataFrame
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
    
    # Get and validate background image
    try:
        # Debug available keys
        logger.debug(f"Available keys: {list(images.keys())}")
        
        # Find background image
        bg_keys = [key for key in images.keys() if "background" in key.lower()]
        if not bg_keys:
            raise ValueError("No background image found in dataset")
        bg_key = bg_keys[0]
        
        # Load background image
        background_pil = images[bg_key]
        logger.debug(f"Background image type: {type(background_pil)}")
        
        if not isinstance(background_pil, Image.Image):
            raise ValueError(f"Background image is not a PIL Image: {type(background_pil)}")
            
        # Convert to numpy array
        background = np.array(background_pil)
        logger.debug(f"Background array shape: {background.shape}, dtype: {background.dtype}")
        
        if background.size == 0:
            raise ValueError("Background image is empty")
            
        if background.ndim < 2:
            raise ValueError(f"Invalid background image dimensions: {background.shape}")
            
        # Crop background to ROI
        background = background[y:y+h, x:x+w]
        logger.debug(f"Background shape after crop: {background.shape}")
        
    except Exception as e:
        logger.error(f"Error processing background image: {str(e)}")
        logger.error("Background image details:")
        if 'bg_key' in locals():
            logger.error(f"Background key: {bg_key}")
            if bg_key in images:
                logger.error(f"Background image type: {type(images[bg_key])}")
        raise
    
    # Process each image
    for key, img_pil in images.items():
        if "background" in key.lower():
            continue
            
        try:
            times = {"image_name": key}
            
            # Convert PIL to numpy and validate
            image = np.array(img_pil)
            if image.ndim < 2:
                logger.warning(f"Skipping {key}: Invalid image dimensions {image.shape}")
                continue
                
            # Debug logging
            logger.debug(f"Processing {key}:")
            logger.debug(f"Original shape: {image.shape}")
            
            # Crop image to ROI
            image = image[y:y+h, x:x+w]
            logger.debug(f"Cropped shape: {image.shape}")
            
            # Ensure images are grayscale
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if len(background.shape) > 2:
                background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
            
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
            
            # Convert back to PIL Image
            processed = processed.astype(np.uint8)
            processed_images[key] = Image.fromarray(processed, mode='L')
            
            # Calculate total time
            times["total_time"] = sum(v for k, v in times.items() if k != "image_name")
            timing_data.append(times)
            
            logger.debug(f"Successfully processed {key}")
            
        except Exception as e:
            logger.error(f"Error processing image {key}: {str(e)}")
            logger.exception("Full traceback:")
            continue
    
    if not processed_images:
        raise ValueError("No images were successfully processed")
    
    timing_df = pd.DataFrame(timing_data)
    numeric_columns = timing_df.select_dtypes(include=['float64']).columns
    timing_df[numeric_columns] = timing_df[numeric_columns].round(2)
    
    logger.info(f"Successfully processed {len(processed_images)} images")
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
    cv_processed: Dict[str, Image.Image],
    reconstructed_sam_masks: Dict[str, Image.Image]
) -> pd.DataFrame:
    """
    Compare OpenCV processed masks with reconstructed SAM masks and calculate metrics.
    
    Args:
        cv_processed: Kedro PartitionedDataSet containing CV processed masks
        reconstructed_sam_masks: Kedro PartitionedDataSet containing SAM masks
        
    Returns:
        DataFrame with comparison metrics (IoU, Dice scores, etc.)
    """
    logger.info("Comparing OpenCV masks with reconstructed SAM masks")
    
    # Convert images to numpy arrays first, making sure to call the load functions
    cv_arrays = {}
    for k, img in cv_processed.items():
        try:
            standardized_key = _standardize_key(k)
            cv_arrays[standardized_key] = np.array(img)
            logger.debug(f"Loaded CV mask {k} with shape {cv_arrays[standardized_key].shape}")
        except Exception as e:
            logger.error(f"Error loading CV image {k}: {str(e)}")
            continue
    
    sam_arrays = {}
    for k, img in reconstructed_sam_masks.items():
        try:
            standardized_key = _standardize_key(k)
            sam_arrays[standardized_key] = np.array(img)
            logger.debug(f"Loaded SAM mask {k} with shape {sam_arrays[standardized_key].shape}")
        except Exception as e:
            logger.error(f"Error loading SAM mask {k}: {str(e)}")
            continue

    # Get common keys
    common_keys = set(cv_arrays.keys()) & set(sam_arrays.keys())
    
    if not common_keys:
        logger.warning("No matching image keys found between CV and SAM datasets")
        logger.warning(f"CV keys: {list(cv_arrays.keys())}")
        logger.warning(f"SAM keys: {list(sam_arrays.keys())}")
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

def create_mask_overlays(
    cv_processed: Dict[str, Image.Image],  # Changed from Dict[str, Callable[[], Any]]
    reconstructed_sam_masks: Dict[str, Image.Image],
    original_images: Dict[str, Image.Image],
    roi: pd.DataFrame
) -> Tuple[Dict[str, Image.Image], Dict[str, Image.Image], Dict[str, Image.Image]]:
    """
    Create overlay visualizations combining original images with masks.
    """
    logger.info("Creating mask overlays")
    
    # Debug logging for keys
    logger.debug(f"CV processed keys: {list(cv_processed.keys())}")
    logger.debug(f"SAM mask keys: {list(reconstructed_sam_masks.keys())}")
    logger.debug(f"Original image keys: {list(original_images.keys())}")
    
    # Standardize keys across all dictionaries
    cv_keys = {_standardize_key(k): k for k in cv_processed.keys()}
    sam_keys = {_standardize_key(k): k for k in reconstructed_sam_masks.keys()}
    orig_keys = {_standardize_key(k): k for k in original_images.keys() 
                 if k != 'background'}
    
    # Find common standardized keys
    common_std_keys = set(cv_keys.keys()) & set(sam_keys.keys()) & set(orig_keys.keys())
    
    logger.info(f"Found {len(common_std_keys)} common keys between datasets")
    if len(common_std_keys) == 0:
        logger.warning("Key matching details:")
        logger.warning(f"Standardized CV keys: {list(cv_keys.keys())}")
        logger.warning(f"Standardized SAM keys: {list(sam_keys.keys())}")
        logger.warning(f"Standardized original keys: {list(orig_keys.keys())}")
    
    OPACITY = 0.2
    COLOR_CV = (255, 0, 0)  # Red
    COLOR_SAM = (0, 255, 0)  # Green
    COLOR_BOTH = (255, 255, 0)  # Yellow
    
    cv_overlays = {}
    sam_overlays = {}
    combined_overlays = {}
    
    # Extract ROI coordinates
    x = roi['x'].iloc[0]
    y = roi['y'].iloc[0]
    w = roi['width'].iloc[0]
    h = roi['height'].iloc[0]
    
    for std_key in sorted(common_std_keys):
        try:
            # Get original keys
            cv_key = cv_keys[std_key]
            sam_key = sam_keys[std_key]
            orig_key = orig_keys[std_key]
            
            # Load all images (now directly accessible, no need to call load function)
            cv_mask = np.array(cv_processed[cv_key])
            sam_mask = np.array(reconstructed_sam_masks[sam_key])
            original = np.array(original_images[orig_key])

            # Crop original image to ROI
            original_cropped = original[y:y+h, x:x+w]

            # Convert masks to binary
            cv_mask = (cv_mask > 127).astype(np.uint8)
            sam_mask = (sam_mask > 127).astype(np.uint8)
            
            # Ensure original image is RGB
            if len(original_cropped.shape) == 2:
                original_cropped = cv2.cvtColor(original_cropped, cv2.COLOR_GRAY2RGB)
            
            # Verify shapes match
            if cv_mask.shape[:2] != original_cropped.shape[:2] or sam_mask.shape[:2] != original_cropped.shape[:2]:
                logger.warning(f"Shape mismatch for {std_key}:")
                logger.warning(f"CV mask: {cv_mask.shape}, SAM mask: {sam_mask.shape}, Original: {original_cropped.shape}")
                continue

            # Create overlays (rest of the overlay creation code remains the same)
            cv_overlay = original_cropped.copy()
            cv_overlay[cv_mask == 1] = cv_overlay[cv_mask == 1] * (1 - OPACITY) + \
                                     np.array(COLOR_CV) * OPACITY
            cv_overlays[std_key] = Image.fromarray(cv_overlay.astype(np.uint8))
            
            sam_overlay = original_cropped.copy()
            sam_overlay[sam_mask == 1] = sam_overlay[sam_mask == 1] * (1 - OPACITY) + \
                                       np.array(COLOR_SAM) * OPACITY
            sam_overlays[std_key] = Image.fromarray(sam_overlay.astype(np.uint8))
            
            combined_overlay = original_cropped.copy()
            combined_overlay[cv_mask == 1] = combined_overlay[cv_mask == 1] * (1 - OPACITY) + \
                                           np.array(COLOR_CV) * OPACITY
            combined_overlay[sam_mask == 1] = combined_overlay[sam_mask == 1] * (1 - OPACITY) + \
                                            np.array(COLOR_SAM) * OPACITY
            overlap = (cv_mask == 1) & (sam_mask == 1)
            combined_overlay[overlap] = combined_overlay[overlap] * (1 - OPACITY) + \
                                     np.array(COLOR_BOTH) * OPACITY
            combined_overlays[std_key] = Image.fromarray(combined_overlay.astype(np.uint8))
            
            logger.debug(f"Created overlays for image {std_key}")
            
        except Exception as e:
            logger.error(f"Error creating overlays for image {std_key}: {str(e)}")
            continue
    
    logger.info(f"Successfully created overlays for {len(common_std_keys)} images")
    return cv_overlays, sam_overlays, combined_overlays