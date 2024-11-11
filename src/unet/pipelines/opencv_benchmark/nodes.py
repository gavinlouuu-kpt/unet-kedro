"""
OpenCV image processing pipeline for contour detection.
"""

import cv2
import numpy as np
from typing import Dict, Any, Callable
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def process_image_partition(
    partition: Dict[str, Callable[[], Any]],
    parameters: Dict[str, Any]
) -> Dict[str, Image.Image]:
    """
    Process all images in a partition.
    
    Args:
        partition: Kedro partition containing images as load functions
        parameters: Parameters containing kernel_size, blur_size, and threshold
    
    Returns:
        Dictionary mapping original partition keys to PIL Images
    """
    logger.info("Processing images in partition")
    logger.debug(f"Parameters: {parameters}")

    # Setup
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, parameters["kernel_size"])
    processed_images = {}
    
    # Get background image from partition
    bg_key = next(key for key in partition.keys() if "background" in key.lower())
    logger.debug(f"Background image dataset: {partition[bg_key]}")
    
    # Load background by calling the load function
    background_pil = partition[bg_key]()  # Call the load function directly
    # Convert PIL Image to numpy array for OpenCV
    background = np.array(background_pil)
    
    if background is None:
        raise ValueError(f"Failed to load background image")
    
    logger.debug(f"Background image shape: {background.shape}")
    
    # Process each image
    for key, load_func in partition.items():
        if "background" in key.lower():
            continue
            
        logger.debug(f"Processing image: {key}")
        # Load image by calling the load function
        image_pil = load_func()  # Call the load function directly
        # Convert PIL Image to numpy array for OpenCV
        image = np.array(image_pil)
        
        if image is None:
            logger.error(f"Failed to load image")
            continue
            
        logger.debug(f"Image shape: {image.shape}")
        
        # Pre-processing steps
        blurred_bg = cv2.GaussianBlur(background, parameters["blur_size"], 0)
        blurred = cv2.GaussianBlur(image, parameters["blur_size"], 0)
        
        bg_sub = cv2.subtract(blurred_bg, blurred)
        _, binary = cv2.threshold(bg_sub, parameters["threshold"], 255, cv2.THRESH_BINARY)
        
        dilate1 = cv2.dilate(binary, kernel, iterations=2)
        erode1 = cv2.erode(dilate1, kernel, iterations=2)
        erode2 = cv2.erode(erode1, kernel, iterations=1)
        processed = cv2.dilate(erode2, kernel, iterations=1)
        
        # Convert to PIL Image
        processed_images[key] = Image.fromarray(processed)
    
    return processed_images