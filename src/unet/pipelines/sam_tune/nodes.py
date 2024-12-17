"""
This is a boilerplate pipeline 'sam_tune'
generated using Kedro 0.19.9
"""
from unet.utils.parse_label_json import LabelParser
from typing import List, Dict, Any, Callable, Tuple
import re
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np
import logging
from transformers import SamProcessor, SamModel
from torch.utils.data import DataLoader, ConcatDataset
from unet.utils.dataset import TuneSAMDataset

from tqdm import tqdm
from statistics import mean
import torch
from torch.optim import Adam
import monai
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

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

     # Log first image and mask sizes
    first_key = common_keys[0]
    first_image = np.array(standardized_images[first_key]().convert('RGB'))
    first_mask = masks[first_key]
    # print(f"Original image shape: {first_image.shape}")
    # print(f"Original mask shape: {first_mask.shape}")

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

def combine_data_collections(data_collections: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Combines multiple data collections into concatenated raw datasets.
    
    Args:
        data_collections: Dictionary where values are methods that return {'train': dict, 'val': dict}
    """
    train_data = []
    val_data = []
    
    for collection_name, load_method in data_collections.items():
        collection_data = load_method()
        train_data.append(collection_data['train'])
        val_data.append(collection_data['val'])
    
    return {
        'train': train_data,
        'val': val_data
    }

def init_processor(params: Dict[str, Any]) -> SamProcessor:
    processor_type = params.get('processor_type', 'facebook/sam-vit-base')
    processor = SamProcessor.from_pretrained(processor_type)
    return processor

def create_dataloaders(
        
    combined_dataset,
    processor,
    params: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader]:
    batch_size = params.get('batch_size', 2)
    
    # Create datasets from raw data
    train_datasets = [TuneSAMDataset(data, processor, prefix='train_') for data in combined_dataset['train']]
    val_datasets = [TuneSAMDataset(data, processor, prefix='val_') for data in combined_dataset['val']]
    
    # Combine datasets
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader




def save_prediction_visualization(image, original_size, reshaped_input_size, pred_mask, gt_mask, batch_idx, epoch, processor):
    """Save visualization of prediction results
    image: original image without any transformations
    original_size: original size of the image
    pred_mask: predicted mask in 256*256 to be postprocessed by the processor in this function
    gt_mask: original ground truth mask
    """
    try:
        
        pred_mask_tensor = pred_mask.unsqueeze(1).float()  # Add missing dimension and to float

        # Move tensors to CPU
        pred_mask_tensor = pred_mask_tensor.cpu().detach()
        original_size = original_size.cpu().detach()
        reshaped_input_size = reshaped_input_size.cpu().detach()
        gt_mask = gt_mask.cpu().detach()
        image = image.cpu().detach()
        
        # Now create the lists for the post-processing function        
        pred_masks_list = [pred_mask_tensor]  # List containing the properly shaped tensor
        original_sizes_list = [[original_size[0].item(), original_size[1].item()]]
        reshaped_input_sizes_list = [[reshaped_input_size[0].item(), reshaped_input_size[1].item()]]

        # Post process the predicted mask
        result = processor.image_processor.post_process_masks(
            masks=pred_masks_list,
            original_sizes=original_sizes_list,
            reshaped_input_sizes=reshaped_input_sizes_list
        )
         # Remove batch and channel dimensions
        image_np = image.squeeze().numpy()           # Shape: (200, 704)
        gt_mask_np = gt_mask.squeeze().numpy()       # Shape: (200, 704)        
        result_np = result[0].squeeze()              # Ensure result is 2D

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Overlay predicted mask
        axes[0].imshow(image_np)
        axes[0].imshow(result_np, cmap='jet', alpha=0.5)
        axes[0].set_title('Predicted Mask Overlay')
        axes[0].axis('off')

        # Overlay ground truth mask
        axes[1].imshow(image_np)
        axes[1].imshow(gt_mask_np, cmap='jet', alpha=0.5)
        axes[1].set_title('Ground Truth Mask Overlay')
        axes[1].axis('off')

        # Overlay both masks
        axes[2].imshow(image_np)
        axes[2].imshow(result_np, cmap='jet', alpha=0.5)
        axes[2].imshow(gt_mask_np, cmap='gray', alpha=0.3)
        axes[2].set_title('Predicted and Ground Truth Overlay')
        axes[2].axis('off')

        # Save with error handling
        save_path = f'data/08_reporting/sam_training/epoch_{epoch}_batch_{batch_idx}.png'
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error saving visualization: {str(e)}")
        plt.close()  # Ensure figure is closed even if save fails

def create_fine_tuned_model(train_dataloader, val_dataloader, processor, parameters: Dict[str, Any]):
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Note: Hyperparameter tuning could improve performance here
    optimizer = Adam(model.mask_decoder.parameters(), lr=parameters.get('learning_rate', 1e-5), weight_decay=parameters.get('weight_decay', 0))
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    num_epochs = parameters.get('num_epochs', 10)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        epoch_losses = []
        for batch_idx, batch in tqdm(enumerate(train_dataloader), desc=f"Training epoch {epoch}"):
            # forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                input_boxes=batch["input_boxes"].to(device),
                multimask_output=False
            )
            
            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Visualize training predictions
            if batch_idx % 10 == 0:
                save_prediction_visualization(
                        image=batch["original_image"][0],
                        original_size=batch["original_sizes"][0],
                        reshaped_input_size=batch["reshaped_input_sizes"][0],
                        pred_mask=predicted_masks[0],
                        gt_mask=batch["original_ground_truth_mask"][0],
                        batch_idx=batch_idx,
                        epoch=epoch,
                        processor=processor
                    )

        train_loss = mean(epoch_losses)
        train_losses.append(train_loss)

        # Validation loop
        model.eval()
        val_epoch_losses = []
        
        with torch.no_grad():  # Disable gradient computation
            for batch_idx, batch in tqdm(enumerate(val_dataloader), desc=f"Validation epoch {epoch}"):
                # forward pass
                outputs = model(
                    pixel_values=batch["pixel_values"].to(device),
                    input_boxes=batch["input_boxes"].to(device),
                    multimask_output=False
                )
                
                # compute loss
                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                val_loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
                val_epoch_losses.append(val_loss.item())

        val_loss = mean(val_epoch_losses)
        val_losses.append(val_loss)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            logger.info(f'New best validation loss: {val_loss:.4f}')

        logger.info(f'EPOCH: {epoch}')
        logger.info(f'Training loss: {train_loss:.4f}')
        logger.info(f'Validation loss: {val_loss:.4f}')

    # Save with huggingface method
    model_save_path = parameters.get('model_save_path', 'data/06_models/sam_bead')
    processor_save_path = parameters.get('processor_save_path', 'data/06_models/sam_bead_processor')
    model.save_pretrained(model_save_path)
    processor.save_pretrained(processor_save_path)

    # Plot losses
    fig = plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.close()

    return best_model_state, fig