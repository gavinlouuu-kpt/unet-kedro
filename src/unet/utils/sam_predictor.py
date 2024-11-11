import torch
from segment_anything import sam_model_registry, SamPredictor
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SAMWrapper:
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize SAM model with given parameters.
        
        Args:
            parameters: Dictionary containing SAM configuration
                - checkpoint_path: Path to model weights
                - model_type: Type of SAM model ('vit_h', 'vit_l', 'vit_b')
                - device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = parameters["sam"]["device"]
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
            
        logger.info(f"Initializing SAM model on {self.device}")
        
        # Initialize model
        sam = sam_model_registry[parameters["sam"]["model_type"]](
            checkpoint=parameters["sam"]["checkpoint_path"]
        )
        sam.to(device=self.device)
        
        self.predictor = SamPredictor(sam)
        logger.info("SAM model initialized successfully")
    
    def set_image(self, image):
        """Set image for prediction"""
        self.predictor.set_image(image)
    
    def predict(self, point_coords=None, point_labels=None, box=None, multimask_output=False):
        """
        Generate mask prediction.
        
        Args:
            point_coords: Optional point coordinates for prompting
            point_labels: Optional point labels for prompting
            box: Optional bounding box for prompting
            multimask_output: Whether to return multiple masks
            
        Returns:
            masks, scores, logits
        """
        return self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output
        )