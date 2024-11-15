from typing import Dict, List, Tuple, Callable, Any
import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from PIL import Image
from .parse_label_json import LabelParser
import cv2
import pandas as pd

logger = logging.getLogger(__name__)

def filter_empty_frames(
    partition: Dict[str, Callable[[], Any]],
    parameters: Dict[str, Any],
    roi: pd.DataFrame
) -> Dict[str, Image.Image]:
    """
    Filter out frames that are empty within the ROI using the same processing pipeline as the main processing.
    
    Args:
        partition: Dictionary of image load functions
        parameters: Processing parameters
        roi: DataFrame containing ROI coordinates (x, y, width, height)
        
    Returns:
        Dictionary of filtered PIL Images (including background)
    """
    logger.info("Filtering empty frames")
    logger.info(f"Initial frame count: {len(partition)}")
    
    # Extract ROI coordinates
    x = roi['x'].iloc[0]
    y = roi['y'].iloc[0]
    w = roi['width'].iloc[0]
    h = roi['height'].iloc[0]
    
    # Setup
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, parameters["kernel_size"])
    filtered_images = {}
    all_areas = []
    
    # Get background image and store as PIL Image
    bg_key = next(key for key in partition.keys() if "background" in key.lower())
    background_pil = partition[bg_key]()
    background = np.array(background_pil)
    background = background[y:y+h, x:x+w]
    filtered_images[bg_key] = background_pil
    
    # Process each image
    for key, load_func in partition.items():
        if "background" in key.lower():
            continue
            
        # Load and process image
        image_pil = load_func()
        image = np.array(image_pil)
        image = image[y:y+h, x:x+w]
        
        # Apply processing
        blurred_bg = cv2.GaussianBlur(background, parameters["blur_size"], 0)
        blurred = cv2.GaussianBlur(image, parameters["blur_size"], 0)
        bg_sub = cv2.subtract(blurred_bg, blurred)
        _, binary = cv2.threshold(bg_sub, parameters["threshold"], 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        dilate1 = cv2.dilate(binary, kernel, iterations=1)
        erode1 = cv2.erode(dilate1, kernel, iterations=1)
        erode2 = cv2.erode(erode1, kernel, iterations=1)
        processed = cv2.dilate(erode2, kernel, iterations=1)
        
        # Calculate area
        white_area = np.sum(processed > 0)
        all_areas.append(white_area)
        
        logger.debug(f"Frame {key}: white area in ROI = {white_area}")
        
        # Keep frame if it has sufficient white pixels in ROI
        if white_area > parameters["minimum_area"]:
            filtered_images[key] = image_pil
            logger.debug(f"Keeping frame {key}")
        else:
            logger.debug(f"Filtering out frame {key}")
    
    # Log statistics
    if all_areas:
        logger.info(f"Area statistics within ROI:")
        logger.info(f"Min area: {min(all_areas)}")
        logger.info(f"Max area: {max(all_areas)}")
        logger.info(f"Mean area: {np.mean(all_areas):.2f}")
        logger.info(f"Median area: {np.median(all_areas):.2f}")
    
    logger.info(f"Filtered {len(partition) - len(filtered_images)} empty frames")
    logger.info(f"Remaining frames: {len(filtered_images)}")
    
    return filtered_images

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


class SegmentationDataset(Dataset):
    def __init__(self, images: Dict, labels: List[Dict]):
        self.images = images
        self.masks = LabelParser.parse_json(labels)
        logger.debug(f"self.images type: {type(self.images)}")
        logger.debug(f"self.images class name: {self.images.__class__.__name__}")
        
        # Extract numbers from keys and create mapping
        def get_number(key: str) -> str:
            # For image.XXXX format
            if key.startswith('image.'):
                return key.split('.')[1]
            # For XXXX.tiff format
            return key.split('.')[0]
        
        # Create mappings from numbers to full keys
        image_number_map = {get_number(k): k for k in self.images.keys()}
        mask_number_map = {get_number(k): k for k in self.masks.keys()}
        
        # Find matching numbers
        image_numbers = set(image_number_map.keys())
        mask_numbers = set(mask_number_map.keys())
        common_numbers = image_numbers.intersection(mask_numbers)
        
        logger.info(f"Number of image keys: {len(self.images)}")
        logger.info(f"Number of mask keys: {len(self.masks)}")
        logger.info(f"Number of matching numbers: {len(common_numbers)}")
        
        # Store matching pairs
        self.image_mask_pairs = [(image_number_map[num], mask_number_map[num]) 
                                for num in common_numbers]
        
        # Log some sample matches
        if self.image_mask_pairs:
            logger.debug("Sample matches (image key -> mask key):")
            for img_key, mask_key in self.image_mask_pairs[:5]:
                logger.debug(f"  {img_key} -> {mask_key}")
        
        # Log unmatched numbers
        unmatched_images = image_numbers - mask_numbers
        unmatched_masks = mask_numbers - image_numbers
        if unmatched_images:
            logger.warning(f"Numbers only in images ({len(unmatched_images)}): {list(unmatched_images)[:5]}")
        if unmatched_masks:
            logger.warning(f"Numbers only in masks ({len(unmatched_masks)}): {list(unmatched_masks)[:5]}")
        
        if not self.image_mask_pairs:
            logger.error("No matching pairs found!")

    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        # Get image and mask keys from the pairs
        image_key, mask_key = self.image_mask_pairs[idx]
        
        # Get raw data
        image = self.images[image_key]
        mask = self.masks[mask_key]
    
        # Handle image data
        if callable(image):
            logger.debug(f"Calling method to get image data for {image_key}")
            image = image()
    
        # Convert to numpy arrays with explicit dtype
        try:
            image = np.asarray(image, dtype=np.float32)
            mask = np.asarray(mask, dtype=np.float32)
            
            logger.debug(f"Image shape: {image.shape}, dtype: {image.dtype}")
            logger.debug(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        except Exception as e:
            logger.error(f"Error converting arrays for key {image_key}: {str(e)}")
            raise
        
        # Convert to torch tensors
        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask)
        
        return image, mask