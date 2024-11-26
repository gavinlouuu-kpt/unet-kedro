# from mobile_sam import sam_model_registry, SamPredictor
from segment_anything import SamPredictor, sam_model_registry
import logging
from typing import Dict, Any, Callable, List, Tuple
import pandas as pd
import re
from PIL import Image
from unet.utils.parse_label_json import LabelParser
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

logger = logging.getLogger(__name__)


def initialize_sam_base(parameters: Dict[str, Any]) -> Any:
    """Initialize SAM model with gradient-enabled wrapper."""
    model_type = parameters["model_type"]
    device = parameters["device"]
    checkpoint = parameters["checkpoint"]

    # Initialize the base model
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    
    # Freeze everything except mask decoder
    for param in sam.parameters():
        param.requires_grad = False
    for param in sam.mask_decoder.parameters():
        param.requires_grad = True
    
    sam.to(device)
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters()) 
    
    return sam, optimizer

def prepare_masks(label_json: List[Dict]) -> Dict[str, np.ndarray]:
    masks = LabelParser.parse_json(label_json)
    
    # Debug first mask
    sample_key = list(masks.keys())[0]
    sample_mask = masks[sample_key]
    logger.info(f"Sample mask shape: {sample_mask.shape}")
    logger.info(f"Sample mask unique values: {np.unique(sample_mask)}")
    logger.info(f"Sample mask min/max: {sample_mask.min()}, {sample_mask.max()}")
    
    logger.info(f"Prepared {len(masks)} masks from Label Studio JSON")
    logger.info(f"Sample keys of masks: {list(masks.keys())[:5]}")
    
    return masks


def prepare_training_data(
    images: Dict[str, Callable[[], Image.Image]],
    masks: Dict[str, np.ndarray],
    params: Dict[str, Any],
    roi: pd.DataFrame,
) -> Dict[str, Any]:
    """Prepare images and masks for SAM fine-tuning."""
    val_split = params.get('val_split', 0.2)
    random_seed = params.get('random_seed', 42)
    
    # Create a mapping of standardized keys
    standardized_images = {
        re.sub(r'image_', '', key): value
        for key, value in images.items()
    }

    # Get sorted list of common keys
    common_keys = sorted(set(standardized_images.keys()) & set(masks.keys()), key=int)
    logger.info(f"Number of matched pairs: {len(common_keys)}")

    # Create matched arrays
    image_dict = {}
    mask_dict = {}

    for key in common_keys:
        image = np.array(standardized_images[key]().convert('RGB'))
        mask = masks[key]
        image_dict[key] = image
        mask_dict[key] = mask

    # Split the keys for train/val
    train_keys, val_keys = train_test_split(
        common_keys,
        test_size=val_split,
        random_state=random_seed
    )

    # Create train/val dictionaries
    train_images = {k: image_dict[k] for k in train_keys}
    train_masks = {k: mask_dict[k] for k in train_keys}
    val_images = {k: image_dict[k] for k in val_keys}
    val_masks = {k: mask_dict[k] for k in val_keys}

    # Extract ROI coordinates
    sample_image = list(image_dict.values())[0]
    box_prompt = np.array([
        max(0, roi['x'].iloc[0]),
        max(0, roi['y'].iloc[0]),
        min(sample_image.shape[1], roi['x'].iloc[0] + roi['width'].iloc[0]),
        min(sample_image.shape[0], roi['y'].iloc[0] + roi['height'].iloc[0])
    ])
    
    logger.info(f"Box prompt coordinates: {box_prompt}")

    return {
        'train': {
            'images': train_images,
            'masks': train_masks,
            'box_prompt': box_prompt
        },
        'val': {
            'images': val_images,
            'masks': val_masks,
            'box_prompt': box_prompt
        }
    }

class SAMDataset(Dataset):
    def __init__(self, images: Dict[str, np.ndarray], masks: Dict[str, np.ndarray], box_prompt: np.ndarray):
        self.image_ids = list(images.keys())
        self.images = images
        self.masks = masks
        self.box_prompt = box_prompt

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        original_image = self.images[image_id]
        h, w = original_image.shape[:2]
        
        # Convert uint8 to float32, normalize to 0-1 range, and resize
        image = torch.from_numpy(original_image).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC -> CHW
        image = F.interpolate(
            image.unsqueeze(0),  # Add batch dimension
            size=(1024, 1024),   # SAM's expected size
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Remove batch dimension
        
        # Also resize the mask
        mask = torch.from_numpy(self.masks[image_id]).float()
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
            size=(1024, 1024),
            mode='nearest'
        ).squeeze(0).squeeze(0)  # Remove extra dimensions
        
        # Scale box coordinates to match 1024x1024
        scale_w = 1024 / w
        scale_h = 1024 / h
        box = self.box_prompt.copy()  # Make a copy to avoid modifying original
        box[0] *= scale_w  # x1
        box[2] *= scale_w  # x2
        box[1] *= scale_h  # y1
        box[3] *= scale_h  # y2
        box = torch.tensor(box).float()
        box = box.unsqueeze(0)  # Add batch dimension [1, 4]
        
        return image, mask, box

def train_sam(
    sam_model: Any,
    optimizer: torch.optim.Optimizer,
    training_data: Dict[str, Any],
    params: Dict[str, Any]
) -> Tuple[Any, Dict[str, List[float]]]:
    """Train SAM model with the provided data."""
    device = params.get('device', 'cuda')
    num_epochs = params.get('num_epochs', 10)
    batch_size = params.get('batch_size', 1)

    train_data = training_data['train']
    val_data = training_data['val']

    # Create datasets and dataloaders
    train_dataset = SAMDataset(train_data['images'], train_data['masks'], train_data['box_prompt'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = SAMDataset(val_data['images'], val_data['masks'], val_data['box_prompt'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    loss_fn = nn.MSELoss()
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        # Training phase
        sam_model.train()
        train_loss = 0.0
        
        for batch_idx, (images, gt_masks, boxes) in enumerate(train_loader):
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            boxes = boxes.to(device)

            # # Add debug logging
            # logger.info(f"Batch shapes:")
            # logger.info(f"Images shape: {images.shape}")
            # logger.info(f"GT masks shape: {gt_masks.shape}")
            # logger.info(f"Boxes shape: {boxes.shape}")

            with torch.no_grad():
                image_embeddings = sam_model.image_encoder(images)
                # logger.info(f"Image embeddings shape: {image_embeddings.shape}")
                
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None,
                )
                # logger.info(f"Sparse embeddings shape: {sparse_embeddings.shape}")
                # logger.info(f"Dense embeddings shape: {dense_embeddings.shape}")
                # logger.info(f"PE shape: {sam_model.prompt_encoder.get_dense_pe().shape}")

            # Generate masks
            try:
                # Get PE and expand to match batch size
                image_pe = sam_model.prompt_encoder.get_dense_pe()
                image_pe = image_pe.expand(batch_size, -1, -1, -1)
                
                # # Log all input shapes
                # logger.info("Shapes going into mask decoder:")
                # logger.info(f"image_embeddings shape: {image_embeddings.shape}")
                # logger.info(f"image_pe shape: {image_pe.shape}")
                # logger.info(f"sparse_prompt_embeddings shape: {sparse_embeddings.shape}")
                # logger.info(f"dense_prompt_embeddings shape: {dense_embeddings.shape}")
                
                low_res_masks, _ = sam_model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings, 
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
            except RuntimeError as e:
                logger.error(f"Error in mask decoder: {str(e)}")
                raise

            # Postprocess masks
            upscaled_masks = sam_model.postprocess_masks(
                low_res_masks, 
                input_size=images.shape[-2:],
                original_size=images.shape[-2:]
            ).to(device)

            binary_masks = F.normalize(F.threshold(upscaled_masks, 0.0, 0))

            # Calculate loss and optimize
            loss = loss_fn(binary_masks, gt_masks.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        sam_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, gt_masks, boxes in val_loader:
                images = images.to(device)
                gt_masks = gt_masks.to(device)
                boxes = boxes.to(device)

                image_embeddings = sam_model.image_encoder(images)
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=boxes,
                    masks=None,
                )

                low_res_masks, _ = sam_model.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                upscaled_masks = sam_model.postprocess_masks(
                    low_res_masks, 
                    input_size=images.shape[-2:],
                    original_size=images.shape[-2:]
                ).to(device)

                binary_masks = F.normalize(F.threshold(upscaled_masks, 0.0, 0))
                loss = loss_fn(binary_masks, gt_masks.unsqueeze(1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {avg_train_loss:.4f} - "
                   f"Val Loss: {avg_val_loss:.4f}")

    return sam_model, history
