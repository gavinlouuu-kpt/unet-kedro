from pathlib import PurePosixPath, Path
from typing import Any, Dict
import numpy as np
from PIL import Image
import fsspec

from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path

from typing import Dict, Any, List, Tuple
import torch
from torch.utils.data import Dataset

# class SAMDataset(Dataset):
#     """Dataset for Segment Anything Model (SAM) training"""
    
#     def __init__(
#         self,
#         images: Dict[str, np.ndarray],
#         masks: Dict[str, np.ndarray],
#         box_prompt: np.ndarray,
#         transform=None
#     ):
#         """
#         Initialize SAM dataset.
        
#         Args:
#             images: Dictionary of images as numpy arrays
#             masks: Dictionary of masks as numpy arrays
#             box_prompt: ROI coordinates [x1, y1, x2, y2]
#             transform: Optional transforms to apply
#         """
#         # Convert dict to sorted lists to ensure consistent ordering
#         self.image_keys = sorted(images.keys())
#         self.images = [images[k] for k in self.image_keys]
#         self.masks = [masks[k] for k in self.image_keys]
        
#         # Convert to torch tensors
#         self.images = [torch.from_numpy(img).float() for img in self.images]
#         self.masks = [torch.from_numpy(mask).float() for mask in self.masks]
#         self.box_prompt = torch.tensor(box_prompt, dtype=torch.float32)
        
#         self.transform = transform

#     def __len__(self) -> int:
#         return len(self.images)

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         image = self.images[idx]
#         mask = self.masks[idx]
        
#         # Convert image from (H,W,C) to (C,H,W)
#         if image.shape[-1] == 3 or image.shape[-1] == 4:
#             image = image.permute(2, 0, 1)
        
#         # If image has alpha channel, convert to RGB
#         if image.shape[0] == 4:
#             image = image[:3, :, :]
            
#         # Ensure mask is 2D
#         if len(mask.shape) > 2:
#             mask = mask.squeeze()
            
#         # Apply transforms if any
#         if self.transform is not None:
#             image = self.transform(image)
#             mask = self.transform(mask)

#         return {
#             'image': image,
#             'mask': mask,
#             'box_prompt': self.box_prompt,
#             'image_id': self.image_keys[idx]
#         }

# def create_sam_dataset(
#     images: Dict[str, np.ndarray],
#     masks: Dict[str, np.ndarray],
#     box_prompt: np.ndarray,
#     params: Dict[str, Any],
#     split: str = 'train'
# ) -> Tuple[SAMDataset, torch.utils.data.DataLoader]:
#     """
#     Create SAM dataset and dataloader.
    
#     Args:
#         images: Dictionary of images
#         masks: Dictionary of masks
#         box_prompt: ROI coordinates
#         params: Training parameters
#         split: Dataset split ('train' or 'val')
    
#     Returns:
#         dataset: SAMDataset instance
#         dataloader: DataLoader instance
#     """
#     # Create dataset
#     dataset = SAMDataset(images, masks, box_prompt)
    
#     # Create dataloader
#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=params.get('batch_size', 4),
#         shuffle=(split == 'train'),
#         num_workers=params.get('num_workers', 2),
#         pin_memory=True
#     )
    
#     return dataset, dataloader

class ImageDataset(AbstractDataset[np.ndarray, np.ndarray]):
    """``ImageDataset`` loads / save image data from a given filepath as `numpy` array using Pillow.

    Example:
    ::

        >>> ImageDataset(filepath='/img/file/path.png')
    """

    def __init__(self, filepath: str):
        """Creates a new instance of ImageDataset to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def load(self) -> np.ndarray:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path, mode="rb") as f:  # Open in binary mode
            image = Image.open(f).convert("RGBA")
            return np.asarray(image)

    def save(self, data: np.ndarray) -> None:
        """Saves image data to the specified filepath."""
        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, mode="wb") as f:  # Open in binary mode
            image = Image.fromarray(data)
            image.save(f)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)