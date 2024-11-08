"""
This is a boilerplate pipeline 'label'
generated using Kedro 0.19.9
"""
from typing import Dict, List
import torch
from torch.utils.data import Dataset
import numpy as np
from unet.utils.parse_label_json import LabelParser
from unet.utils.dataset import SegmentationDataset
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

# TODO: use the label_studio_sdk.converter.brush to decode the rle data


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

