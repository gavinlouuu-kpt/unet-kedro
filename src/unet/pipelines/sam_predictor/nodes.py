"""
This is a boilerplate pipeline 'sam_predictor'
generated using Kedro 0.19.9
"""

# example usage from github
# sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
# predictor = SamPredictor(sam)
# predictor.set_image(<your_image>)
# masks, _, _ = predictor.predict(<input_prompts>)

from mobile_sam import sam_model_registry, SamPredictor
import logging
from typing import Dict, Any, Callable
import pandas as pd
import re
from PIL import Image
from unet.utils.dataset import _standardize_key
import numpy as np
import time
from pathlib import Path

logger = logging.getLogger(__name__)

def initialize_sam(parameters: Dict[str, Any]) -> SamPredictor:
    """
    Initialize SAM model using checkpoint from catalog.
    
    Args:
        sam_checkpoint: Model weights loaded from catalog
        parameters: Parameters containing model configuration
            Expected format:
            {
                "sam": {
                    "model_type": "vit_h",  # or "vit_l", "vit_b"
                    "device": "cuda"  # or "cpu"
                }
            }
    Returns:
        SamPredictor: Initialized SAM predictor
    """
    logger.info("Initializing SAM model")
    
    # Get parameters
    model_type = parameters["model_type"]
    device = parameters["device"]
    checkpoint_path = parameters["checkpoint"]
    
    # Initialize model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    # Create predictor
    predictor = SamPredictor(sam)
    
    logger.info(f"SAM model initialized successfully on {device}")
    return predictor

def _get_image_number(filename: str) -> str:
    """
    Extract last 4-digit sequence from filename.
    Example:
    - f006418c-0078.png -> 0078
    - image.0059.png -> 0059
    """
    # Find all sequences of digits in the filename
    numbers = re.findall(r'\d{4}', filename)
    if numbers:
        # Take the last 4-digit sequence found
        return numbers[-1]
    logger.warning(f"No 4-digit sequence found in filename: {filename}")
    return None

def prepare_cropped_images(
    partition: Dict[str, Callable[[], Any]],
    roi: pd.DataFrame
) -> Dict[str, Image.Image]:
    """
    Prepare cropped images from partition for SAM prediction.
    
    Args:
        partition: Kedro partition containing images as load functions
        roi: DataFrame containing ROI coordinates (x, y, width, height)
        
    Returns:
        Dictionary mapping image keys to cropped PIL Images
    """
    logger.info("Preparing cropped images for SAM prediction")
    
    # Extract ROI coordinates
    x = roi['x'].iloc[0]
    y = roi['y'].iloc[0]
    w = roi['width'].iloc[0]
    h = roi['height'].iloc[0]
    
    cropped_images = {}
    
    for key, load_func in partition.items():
        # Skip background image
        if "background" in key.lower():
            continue
            
        # Load and crop image
        image_pil = load_func()
        # Crop using PIL for better handling of image formats
        cropped = image_pil.crop((x, y, x+w, y+h))
        
        # Store with standardized key
        std_key = _standardize_key(key)
        cropped_images[std_key] = cropped
        
    logger.info(f"Prepared {len(cropped_images)} cropped images")
    return cropped_images

def predict_masks(
    predictor: SamPredictor,
    cropped_images: Dict[str, Callable[[], Image.Image]]
) -> Dict[str, Any]:
    """
    Predict masks using SAM with full image rectangle as prompt.
    
    Args:
        predictor: Initialized SAM predictor
        cropped_images: Kedro partition containing images as load functions
        
    Returns:
        Dictionary with two keys:
            'masks': Dict[str, Image.Image] - mapping partition keys to binary mask images
            'timing': pd.DataFrame - DataFrame with image names and processing times in milliseconds
    """
    logger.info("Predicting masks using SAM")
    
    results = {}
    timing_data = []
    
    for key, load_func in cropped_images.items():
        try:
            # Convert to milliseconds by multiplying by 1000
            start_time = time.time() * 1000
            
            # Load image
            image = load_func()
            # Convert PIL image to RGB numpy array (required by SAM)
            image_array = np.array(image.convert('RGB'))
            
            # Set image in predictor
            predictor.set_image(image_array)
            
            # Get image dimensions for rectangle prompt
            height, width = image_array.shape[:2]
            
            # Create box prompt using entire image dimensions
            input_box = np.array([0, 0, width, height])
            
            # Predict single mask
            masks, _, _ = predictor.predict(
                box=input_box[None, :],  # Add batch dimension
                multimask_output=False
            )
            
            # Convert binary mask to PIL Image (grayscale)
            mask = masks[0].astype(np.uint8) * 255
            mask_image = Image.fromarray(mask, mode='L')
            
            # Calculate processing time in milliseconds
            end_time = time.time() * 1000
            processing_time = end_time - start_time
            
            # Store mask
            std_key = _standardize_key(key)
            results[std_key] = mask_image
            
            # Store timing data with ms
            image_name = Path(key).stem
            timing_data.append({
                'image_name': image_name,
                'processing_time_ms': processing_time
            })
            
            logger.debug(f"Successfully processed image {key} in {processing_time:.2f} ms")
            
        except Exception as e:
            logger.error(f"Error processing image {key}: {str(e)}")
            raise
        
    # Create DataFrame with timing data
    timing_df = pd.DataFrame(timing_data)
    
    logger.info(f"Completed mask prediction for {len(results)} images")
    logger.info(f"Average processing time: {timing_df['processing_time_ms'].mean():.2f} ms")
    
    return {
        'masks': results,
        'timing': timing_df
    }