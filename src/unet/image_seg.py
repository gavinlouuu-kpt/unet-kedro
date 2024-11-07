from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from typing import Dict, List
import numpy as np
import re

bootstrap_project(Path.cwd())

class LabelParser:
    @staticmethod
    def _get_image_number(filename: str) -> str:
        """Extract image number from filename regardless of extension"""
        # Extract number from patterns like 'image.0059.png' or '3bf67bb8-image.0059.png'
        match = re.search(r'\.(\d+)\.[^.]+$', filename)
        if match:
            return f"{int(match.group(1)):04d}.tiff"
        return None

    @staticmethod
    def decode_rle(rle: List[int], shape: tuple) -> np.ndarray:
        """Decode run-length encoded data into a binary mask"""
        mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        position = 0
        for i in range(0, len(rle), 2):
            position += rle[i]
            if i + 1 < len(rle):
                mask[position:position + rle[i + 1]] = 1
                position += rle[i + 1]
        return mask.reshape(shape)

    @staticmethod
    def parse_json(json_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Parse the Label Studio JSON format into image_name: mask pairs"""
        masks = {}
        for item in json_data:
            # Get the file upload name and convert to tiff format
            file_upload = item['file_upload']  # e.g., "3bf67bb8-image.0059.png"
            tiff_name = LabelParser._get_image_number(file_upload)
            
            if not tiff_name:
                continue
                
            # Process annotations
            for annotation in item['annotations']:
                for result in annotation['result']:
                    if result.get('type') == 'brushlabels':
                        # Get image dimensions
                        height = result['original_height']
                        width = result['original_width']
                        
                        # Decode RLE data
                        rle_data = result['value']['rle']
                        mask = LabelParser.decode_rle(rle_data, (height, width))
                        
                        # Store with TIFF filename
                        masks[tiff_name] = mask
        
        return masks

class SegmentationDataset(Dataset):
    def __init__(self, images: Dict, labels: Dict):
        self.images = images
        self.masks = LabelParser.parse_json(labels)
        self.image_files = [img for img in images.keys() if img in self.masks]

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = self.images[img_name]
        mask = self.masks[img_name]
        
        # Convert to torch tensors and add necessary preprocessing
        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask)
        
        return image, mask

def image_segmentation():
    with KedroSession.create() as session:
        context = session.load_context()
        
        # Load data from catalog
        images = context.catalog.load("In_focus_partition")
        labels = context.catalog.load("segmentation_labels")
        unet_params = context.params["params:unet"]
        
        # Create dataset
        dataset = SegmentationDataset(images, labels)
        
        # Save the prepared dataset
        context.catalog.save("segmentation_dataset", dataset)
        
        # You can add model training here or in a separate node
        # model = train_model(dataset, unet_params)
        # context.catalog.save("trained_unet", model)

if __name__ == "__main__":
    image_segmentation()
