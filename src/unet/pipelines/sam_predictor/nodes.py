"""
This is a boilerplate pipeline 'sam_predictor'
generated using Kedro 0.19.9
"""

# def initialize_sam(parameters: Dict[str, Any]) -> SAMWrapper:
#     """Initialize SAM model with given parameters"""
#     return SAMWrapper(parameters)

# def predict_mask(
#     sam_model: SAMWrapper,
#     image: np.ndarray,
#     prompt_points: np.ndarray = None,
#     prompt_box: np.ndarray = None
# ) -> np.ndarray:
#     """
#     Generate mask prediction using SAM.
    
#     Args:
#         sam_model: Initialized SAM model
#         image: Input image
#         prompt_points: Optional point prompts
#         prompt_box: Optional box prompt
        
#     Returns:
#         Binary mask
#     """
#     # Set image
#     sam_model.set_image(image)
    
#     # Get prediction
#     masks, scores, _ = sam_model.predict(
#         point_coords=prompt_points,
#         point_labels=None if prompt_points is None else np.ones(len(prompt_points)),
#         box=prompt_box,
#         multimask_output=False
#     )
    
#     # Return highest scoring mask
#     return masks[0]


# example usage from github
# sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
# predictor = SamPredictor(sam)
# predictor.set_image(<your_image>)
# masks, _, _ = predictor.predict(<input_prompts>)

from segment_anything import sam_model_registry, SamPredictor
import logging
from typing import Dict, Any

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