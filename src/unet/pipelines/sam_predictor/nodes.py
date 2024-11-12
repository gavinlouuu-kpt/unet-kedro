"""
This is a boilerplate pipeline 'sam_predictor'
generated using Kedro 0.19.9
"""

# example usage from github
# sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
# predictor = SamPredictor(sam)
# predictor.set_image(<your_image>)
# masks, _, _ = predictor.predict(<input_prompts>)

from segment_anything import sam_model_registry, SamPredictor
import logging
from typing import Dict, Any, Callable
import pandas as pd
import re
from PIL import Image
from unet.utils.dataset import _standardize_key

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