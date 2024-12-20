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
        Issue: when mixing dataset there could be repeated image name.
        Extract number between underscore/dot and extension dot in filename.
        Example:
        - e054e7f2-image_6.png -> 6
        - image_123.jpg -> 123
        - test_45.jpeg -> 45
        - image.456.png -> 456
        - sample.789.jpg -> 789
        """
        match = re.search(r'[_.](\d+)\.', filename)
        if match:
            return match.group(1)
        logger.warning(f"No number found between underscore/dot and extension in filename: {filename}")
        return None

    @staticmethod
    def parse_json(json_data: List[Dict], parse_by_image_number: bool = False) -> Dict[str, np.ndarray]:
        """Parse the Label Studio JSON format into image_name: mask pairs"""
        masks = {}
        logger.info(f"Processing {len(json_data)} total instances from Label Studio JSON")
        
        for item in json_data:
            # Get the file upload name and convert to tiff format
            file_upload = item['file_upload']
            if parse_by_image_number:
                tiff_name = LabelParser._get_image_number(file_upload)
            else:
                tiff_name = file_upload
            
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