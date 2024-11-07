from typing import List, Dict
import numpy as np
import re

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
                        
                        # Get label type (Single Cell or Cluster)
                        label_type = result['value']['brushlabels'][0]
                        # You might want to handle different label types differently
                        # For now, we're just creating binary masks
        
        return masks