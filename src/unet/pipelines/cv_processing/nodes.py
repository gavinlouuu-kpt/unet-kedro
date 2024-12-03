"""
This is a boilerplate pipeline 'cv_processing'
generated using Kedro 0.19.9
"""

import cv2
import numpy as np
from typing import Dict, Any, Callable
from PIL import Image
import json
import logging
from torch.utils.data import DataLoader
from unet.utils.dataset import PrepareOpenCVDataset, OpenCVInference

logger = logging.getLogger(__name__)


def create_cv_data_dict(image, roi) -> Dict[str, Any]:
    cv_data_dict = {
        'images': image, 
        'roi': roi
    }
    return cv_data_dict

def prepare_cv_dataset(data_dict: Dict[str, Any], config_json):
    dataset = PrepareOpenCVDataset(data_dict['images'], data_dict['roi'], config_json)
    return dataset

def run_processing(dataset, config_json):
    model = OpenCVInference(config_json)
    predictions = model.perform_processing(dataset)
    return predictions

def img_process(images, config_json, roi):
    def ensure_tuple(value):
        if isinstance(value, int):
            return (value, value)
        return tuple(value)

    x = roi['x'].iloc[0]
    y = roi['y'].iloc[0]
    w = roi['width'].iloc[0]
    h = roi['height'].iloc[0]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS, ensure_tuple(config_json["morph_kernel_size"])
    )
    processed_images = {}

    # Find background image
    bg_keys = [key for key in images.keys() if "background_clean" in key.lower()]
    if not bg_keys:
        raise ValueError("No background image found in dataset")
    bg_key = bg_keys[0]

    # Load the background image by calling the function
    background_dataset = images[bg_key]
    background = background_dataset()
    logger.info(f"Type of background: {type(background)}")

    # Check if the background image is correctly loaded
    if not isinstance(background, Image.Image):
        raise ValueError(f"Background image {bg_key} is not a valid PIL image")

    background = np.array(background)

    # Check if the background image is correctly converted to a NumPy array
    if background.ndim == 0:
        raise ValueError(
            f"Background image {bg_key} could not be converted to a valid NumPy array"
        )

    # Crop background to ROI
    background = background[y : y + h, x : x + w]

    for key, img_load_func in images.items():
        if "background_clean" in key.lower():
            continue

        # Load each image by calling the function
        img = img_load_func()
        if not isinstance(img, Image.Image):
            logger.warning(f"Image {key} is not a valid PIL image and will be skipped.")
            continue

        image = np.array(img)

        # Crop image to ROI
        image = image[y : y + h, x : x + w]

        # Apply image processing steps
        blurred_bg = cv2.GaussianBlur(
            background, ensure_tuple(config_json["gaussian_blur_size"]), 0
        )
        blurred_targ = cv2.GaussianBlur(
            image, ensure_tuple(config_json["gaussian_blur_size"]), 0
        )
        bg_sub = cv2.subtract(blurred_bg, blurred_targ)
        _, binary = cv2.threshold(
            bg_sub, config_json["bg_subtract_threshold"], 255, cv2.THRESH_BINARY
        )
        dilate1 = cv2.dilate(binary, kernel, iterations=1)
        erode1 = cv2.erode(dilate1, kernel, iterations=1)
        erode2 = cv2.erode(erode1, kernel, iterations=1)
        processed = cv2.dilate(erode2, kernel, iterations=1)

        processed = processed.astype(np.uint8)
        processed_images[key] = Image.fromarray(processed, mode="L")

    return processed_images

def contour_process_cv(processed_images):
    """
    Processes a dictionary containing image masks and generates statistics.
    
    Parameters:
    processed_images (dict): Dictionary containing image data with masks.
    
    Returns:
    dict: Dictionary with added statistics for each mask.
    """
    for key, result in processed_images.items():
        contours_info = []
        for mask in result['masks']:
            if hasattr(mask, 'numpy'):
                # Convert mask to numpy array if it's not already
                mask = mask.numpy()
            
            if mask.ndim == 4:
                mask = mask.squeeze(0).squeeze(0)
            
            mask = mask.astype(np.uint8)
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Check if there is more than one contour
            if len(contours) > 1:
                result['DI'] = None
                break

            for contour in contours:
                # Calculate area
                area = cv2.contourArea(contour)
                
                # Calculate perimeter
                perimeter = cv2.arcLength(contour, True)
                
                # Calculate deformability (perimeter to area ratio)
                deformability = perimeter / area if area != 0 else 0
                
                convex_hull = cv2.convexHull(contour)

                # Calculate convex hull area
                convex_hull_area = cv2.contourArea(convex_hull)
                
                # Calculate area ratio between convex hull and contour
                area_ratio = convex_hull_area / area if area != 0 else 0

                contours_info.append({
                    'contour': contour,
                    'area': area,
                    'deformability': deformability,
                    'area_ratio': area_ratio
                })
        
        result['DI'] = contours_info

    return processed_images