"""
OpenCV image processing pipeline for contour detection.
"""

import cv2
import numpy as np
from typing import Dict, Any, Callable, Tuple
from PIL import Image
import logging
import time
import pandas as pd

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
        
        # Calculate total time
        times["total_time"] = sum(v for k, v in times.items() if k != "image_name")
        
        # Add to timing data
        timing_data.append(times)
        
        # Convert to PIL Image
        processed_images[key] = Image.fromarray(processed)
    
    timing_df = pd.DataFrame(timing_data)
    
    # Round all numeric columns to 2 decimal places
    numeric_columns = timing_df.select_dtypes(include=['float64']).columns
    timing_df[numeric_columns] = timing_df[numeric_columns].round(2)
    
    return processed_images, timing_df