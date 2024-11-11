from typing import List, Dict
import numpy as np
import re
from label_studio_sdk.converter.brush import decode_from_annotation
import logging

logger = logging.getLogger(__name__)

class LabelParser:
    @staticmethod
    def _get_image_number(filename: str) -> str:
        """
        Extract last 4-digit sequence from filename.
        Example:
        - f006418c-0078.png -> 0078
        - image.0059.png -> 0059
        """
        # Find all sequences of digits in the filename
        numbers = re.findall(r'\d{4}', filename)
        if numbers:
            # Take the last 4-digit sequence found
            return numbers[-1]  # Changed from [0] to [-1]
        logger.warning(f"No 4-digit sequence found in filename: {filename}")
        return None

    @staticmethod
    def parse_json(json_data: List[Dict]) -> Dict[str, np.ndarray]:
        """Parse the Label Studio JSON format into image_name: mask pairs"""
        masks = {}
        logger.info(f"Processing {len(json_data)} total instances from Label Studio JSON")
        
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
                        
                        try:
                            # Decode using label-studio-sdk
                            layers = decode_from_annotation('image', formatted_result)
                            
                            # Store the first layer (assuming single class)
                            for _, mask in layers.items():
                                # Ensure mask is binary (0 or 1)
                                binary_mask = (mask > 0).astype(np.uint8)
                                masks[tiff_name] = binary_mask
                                break  # Only take the first layer
                        except Exception as e:
                            logger.error(f"Error processing mask for {tiff_name}: {str(e)}")
                            continue
        
        logger.info(f"Successfully parsed {len(masks)} masks from Label Studio JSON")
        return masks