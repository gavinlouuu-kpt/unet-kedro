"""
This is a boilerplate pipeline 'sam_inference'
generated using Kedro 0.19.9
"""
from typing import Any, Dict
from transformers import SamProcessor
from unet.utils.dataset import PrepareSAMDataset, SamInference
from torch.utils.data import DataLoader
from pathlib import Path
import cv2
import logging

logger = logging.getLogger(__name__)

def create_data_dict(image, roi) -> Dict[str, Any]:
    data_dict = {
        'images': image, 
        'box_prompt': roi
    }
    return data_dict

def initSAM(parameters: Dict[str, Any]):
    processor_path = parameters['processor']
    model_path = parameters['model']

    logger = logging.getLogger(__name__)

    if not Path(processor_path).exists():
        logger.error(f"Processor path does not exist: {processor_path}")
        raise FileNotFoundError(f"Processor path does not exist: {processor_path}")

    if not Path(model_path).exists():
        logger.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    processor = SamProcessor.from_pretrained(processor_path)
    model = SamInference(model_path, processor_path)
    return processor, model


def prepare_dataloader(data_dict: Dict[str, Any], processor) -> Dict[str, Any]:
    dataset = PrepareSAMDataset(data_dict['images'], data_dict['box_prompt'], processor)
    dataloader = DataLoader(dataset, batch_size=1)
    return dataloader

def run_inference(dataloader: Any, model):
    predictions = model.perform_inference(dataloader)
    return predictions


def process_masks(pred_pair):
    for result in pred_pair:
        contours_info = []
        for mask in result['masks']:
            # Convert mask to numpy array if it's not already
            mask = mask.cpu().numpy()
            
            if mask.ndim == 4:
                mask = mask.squeeze(0).squeeze(0)
            
            mask = mask.astype(np.uint8)
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Check if there is more than one contour
            if len(contours) > 1:
                result['DI'] = None
                break

            for contour in contours:
                # Calculate area
                area = cv2.contourArea(contour)
                
                # Calculate perimeter
                perimeter = cv2.arcLength(contour, True)
                
                # Calculate deformability (perimeter to area ratio)
                deformability = perimeter / area if area != 0 else 0
                
                convex_hull = cv2.convexHull(contour)

                # Calculate convex hull area
                convex_hull_area = cv2.contourArea(convex_hull)
                
                # Calculate area ratio between convex hull and contour
                area_ratio = convex_hull_area / area if area != 0 else 0

                contours_info.append({
                    'contour': contour,
                    'area': area,
                    'deformability': deformability,
                    'area_ratio': area_ratio
                })
        
        result['DI'] = contours_info

    return pred_pair

def processingDI(predictions):
    predictions_DI = process_masks(predictions)
    return predictions_DI


