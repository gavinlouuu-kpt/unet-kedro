from typing import List, Dict
import numpy as np
import re
from label_studio_sdk.converter.brush import decode_from_annotation

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
    def parse_json(json_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Parse the Label Studio JSON format into image_name: mask pairs"""
        masks = {}
        
        for item in json_data:
            # Get the file upload name and convert to tiff format
            file_upload = item['file_upload']
            tiff_name = LabelParser._get_image_number(file_upload)
            
            if not tiff_name or not item.get('annotations'):
                continue
            
            # Process annotations
            for annotation in item['annotations']:
                for result in annotation['result']:
                    if result.get('type') == 'brushlabels':
                        # Format result for the decoder
                        formatted_result = [{
                            'type': 'brushlabels',
                            'rle': result['value']['rle'],
                            'original_width': result['original_width'],
                            'original_height': result['original_height'],
                            'brushlabels': result['value']['brushlabels']
                        }]
                        
                        # Decode using label-studio-sdk
                        layers = decode_from_annotation('image', formatted_result)
                        
                        # Store the first layer (assuming single class)
                        for _, mask in layers.items():
                            masks[tiff_name] = mask.astype(np.uint8)
                            break  # Only take the first layer
        
        return masks