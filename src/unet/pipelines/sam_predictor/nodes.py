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
from typing import Dict, Any, Tuple, List
import numpy as np
import re

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

def parse_label_studio_json(json_data: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Parse Label Studio JSON annotations to extract prompt points.
    
    Args:
        json_data: List of Label Studio annotation dictionaries
        
    Returns:
        Dictionary mapping image numbers to prompt points array
        Format: {
            "0078": np.array([[x1, y1], [x2, y2], ...])
        }
    """
    logger.info(f"Processing {len(json_data)} annotations from Label Studio")
    prompt_points = {}
    
    for item in json_data:
        try:
            # Get filename and extract number
            filename = item['file_upload']
            image_number = _get_image_number(filename)
            
            if not image_number:
                logger.warning(f"Could not extract image number from: {filename}")
                continue
            
            # Process annotations
            if not item.get('annotations'):
                logger.warning(f"No annotations found for image {image_number}")
                continue
                
            points = []
            for annotation in item['annotations']:
                for result in annotation['result']:
                    # Check for keypoint/point annotations
                    if result.get('type') == 'keypointlabels':
                        x = result['value']['x']
                        y = result['value']['y']
                        points.append([x, y])
            
            if points:
                prompt_points[image_number] = np.array(points)
                logger.debug(f"Extracted {len(points)} points for image {image_number}")
            else:
                logger.warning(f"No valid points found in annotations for image {image_number}")
                
        except Exception as e:
            logger.error(f"Error processing annotation: {str(e)}")
            continue
    
    logger.info(f"Successfully extracted prompt points for {len(prompt_points)} images")
    return prompt_points
