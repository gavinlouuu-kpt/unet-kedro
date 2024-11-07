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

def create_segmentation_dataset(images: Dict, labels: List[Dict]) -> Dataset:
    """Creates a PyTorch Dataset for image segmentation"""
    logger.info(f"Creating segmentation dataset with {len(images)} images and {len(labels)} labels")
    dataset = SegmentationDataset(images, labels)
    
    # Test first item access
    try:
        logger.debug("Testing first item access...")
        image, mask = dataset[0]
        logger.debug("Successfully loaded first item")
    except Exception as e:
        logger.error(f"Error accessing first item: {str(e)}")
        # Print the first image and mask data
        image_key, mask_key = dataset.image_mask_pairs[0]
        logger.debug(f"First image key: {image_key}")
        logger.debug(f"First image data type: {type(dataset.images[image_key])}")
        if hasattr(dataset.images[image_key], 'shape'):
            logger.debug(f"First image shape: {dataset.images[image_key].shape}")
        logger.debug(f"First mask key: {mask_key}")
        logger.debug(f"First mask data type: {type(dataset.masks[mask_key])}")
        if hasattr(dataset.masks[mask_key], 'shape'):
            logger.debug(f"First mask shape: {dataset.masks[mask_key].shape}")
        raise
    
    return dataset
