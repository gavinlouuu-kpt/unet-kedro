"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.9
"""
from typing import Dict, List, Tuple
import torch
import logging
from unet.utils import SegmentationDataset


logger = logging.getLogger(__name__)

def test_node(Dataset):
    logger.info("Test node executed")
    

def split_data(dataset: SegmentationDataset, split_ratio: float) -> Tuple[SegmentationDataset, SegmentationDataset]:
    """Split the dataset into training and validation sets"""
    train_size = int(len(dataset) * (1 - split_ratio))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    return train_dataset, val_dataset

# # model creation
# def create_model(in_channels: int, out_channels: int) -> nn.Module:
#     """Create a U-Net model"""
#     return UNet(in_channels, out_channels)

# # model training
# def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int, learning_rate: float) -> nn.Module:
#     """Train the model"""
#     return train_model(model, train_loader, val_loader, num_epochs, learning_rate)
