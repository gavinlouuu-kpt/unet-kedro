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
    learning_rate = parameters["learning_rate"]
    # Initialize the base model
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    
    # # Freeze everything except mask decoder
    # for param in sam.parameters():
    #     param.requires_grad = False
    # for param in sam.mask_decoder.parameters():
    #     param.requires_grad = True
    
    sam.to(device)
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters(), lr=learning_rate) 
    
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

def visualize_prediction(image, pred_mask, gt_mask, epoch, batch_idx, save_dir):
    """Visualize the prediction, ground truth, and overlay."""
    plt.figure(figsize=(15, 5))
    
    # Convert image from CHW to HWC format and denormalize
    image_np = image.cpu().permute(1, 2, 0).numpy()
    pred_mask_np = pred_mask.cpu().squeeze().detach().numpy()
    gt_mask_np = gt_mask.cpu().detach().numpy()
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot prediction
    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask_np, cmap='gray')
    plt.title('Prediction')
    plt.axis('off')
    
    # Plot ground truth
    plt.subplot(1, 3, 3)
    plt.imshow(gt_mask_np, cmap='gray')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Save the figure
    save_path = Path(save_dir) / f'epoch_{epoch}_batch_{batch_idx}.png'
    plt.savefig(save_path)
    plt.close()

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
    batch_losses = []

    train_data = training_data['train']
    val_data = training_data['val']

    # Create datasets and dataloaders
    train_dataset = SAMDataset(train_data['images'], train_data['masks'], train_data['box_prompt'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = SAMDataset(val_data['images'], val_data['masks'], val_data['box_prompt'])
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    loss_fn = nn.BCEWithLogitsLoss()
    history = {'train_loss': [], 'val_loss': []}

    # Add visualization parameters
    vis_frequency = params.get('vis_frequency', 50)  # Visualize every 50 batches
    vis_dir = Path(params.get('vis_dir', 'visualization_output'))
    vis_dir.mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        sam_model.train()
        train_loss = 0.0
        num_batches = 0  # Add counter for proper averaging
        
        for batch_idx, (images, gt_masks, boxes) in enumerate(train_loader):
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            boxes = boxes.to(device)

            # Zero gradients at start of each batch
            optimizer.zero_grad()

            # Remove the torch.no_grad() block - we need gradients for training
            image_embeddings = sam_model.image_encoder(images)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,
            )

            # Get PE and expand to match batch size
            image_pe = sam_model.prompt_encoder.get_dense_pe()
            image_pe = image_pe.expand(batch_size, -1, -1, -1)
            
            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings, 
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # Postprocess masks
            upscaled_masks = sam_model.postprocess_masks(
                low_res_masks, 
                input_size=images.shape[-2:],
                original_size=images.shape[-2:]
            ).to(device)

            # Calculate loss directly with BCEWithLogitsLoss
            loss = loss_fn(upscaled_masks, gt_masks.unsqueeze(1))
            batch_losses.append(loss.item())
             # Proper loss accumulation
            train_loss += loss.item()
            num_batches += 1
            # Backward and optimize
            loss.backward()

            # Check gradients for first batch of each epoch
            if batch_idx == 0:  # Check first batch of each epoch
                logger.info(f"Epoch {epoch} gradients:")
                for name, param in sam_model.mask_decoder.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        logger.info(f"{name}: grad_norm = {grad_norm}")
                        if grad_norm == 0:
                            logger.warning(f"Zero gradient for {name}!")
                            
                with torch.no_grad():
                    pred_unique = torch.sigmoid(upscaled_masks[0]).unique()
                    gt_unique = gt_masks[0].unique()
                    logger.info(f"Epoch {epoch} - Prediction values: {pred_unique}")
                    logger.info(f"Epoch {epoch} - Ground truth values: {gt_unique}")
                    logger.info(f"Prediction mean: {torch.sigmoid(upscaled_masks[0]).mean():.4f}")
                    logger.info(f"Ground truth mean: {gt_masks[0].mean():.4f}")

            optimizer.step()

            # log the loss every 10 batches with the epoch and batch index
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx} - Loss: {loss.item()}")

            # Add debugging prints
            if batch_idx == 0:
                logger.info(f"Loss value: {loss.item()}")
                # logger.info(f"Grad norms: {[p.grad.norm().item() for p in sam_model.mask_decoder.parameters() if p.grad is not None]}")

            # Add visualization
            if batch_idx % vis_frequency == 0:
                visualize_prediction(
                    images[0],  # Take first image from batch
                    torch.sigmoid(upscaled_masks[0]),  # Take first prediction and apply sigmoid for visualization
                    gt_masks[0],  # Take first ground truth
                    epoch,
                    batch_idx,
                    vis_dir
                )

        # Correct averaging
        avg_train_loss = train_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        # Add debugging prints
        logger.info(f"Raw train_loss: {train_loss}, num_batches: {num_batches}")
        # logger.info(f"Train batch losses: {batch_losses}")

        # Validation phase
        sam_model.eval()
        val_loss = 0.0
        val_batch_losses = []
        num_val_batches = 0  # Add counter for validation too
        
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

                loss = loss_fn(upscaled_masks, gt_masks.unsqueeze(1))
                val_loss += loss.item()
                val_batch_losses.append(loss.item())
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        logger.info(f"Raw val_loss: {val_loss}, num_val_batches: {num_val_batches}")
        # logger.info(f"Val batch losses: {val_batch_losses}")
        history['val_loss'].append(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {avg_train_loss:.4f} - "
                   f"Val Loss: {avg_val_loss:.4f}")

    return sam_model, history
