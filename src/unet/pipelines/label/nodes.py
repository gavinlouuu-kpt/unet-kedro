"""
This is a boilerplate pipeline 'label'
generated using Kedro 0.19.9
"""
from typing import Dict, List
import torch
from torch.utils.data import Dataset
import numpy as np
from unet.utils.parse_label_json import LabelParser
import logging

# Get logger for this module
logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    def __init__(self, images: Dict, labels: List[Dict]):
        self.images = images
        self.masks = LabelParser.parse_json(labels)
        
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
        
        # Debug prints
        logger.debug(f"Image key: {image_key}")
        logger.debug(f"Image type: {type(self.images[image_key])}")
        logger.debug(f"Mask key: {mask_key}")
        logger.debug(f"Mask type: {type(self.masks[mask_key])}")
        
        image = np.asarray(self.images[image_key])
        mask = np.asarray(self.masks[mask_key])
        
        if callable(image):
            raise TypeError(f"Image data is a method: {image}")
        if callable(mask):
            raise TypeError(f"Mask data is a method: {mask}")
        
        # Convert to torch tensors
        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask)
        
        return image, mask

def create_segmentation_dataset(images: Dict, labels: List[Dict]) -> Dataset:
    """Creates a PyTorch Dataset for image segmentation"""
    logger.info(f"Creating segmentation dataset with {len(images)} images and {len(labels)} labels")
    return SegmentationDataset(images, labels)
