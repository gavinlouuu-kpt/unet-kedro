"""
This is a boilerplate pipeline 'sam_inference'
generated using Kedro 0.19.9
"""
from typing import Any, Dict
from transformers import SamProcessor
from unet.utils.dataset import PrepareSAMDataset, SamInference
from torch.utils.data import DataLoader
from pathlib import Path

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

def processingDI(predictions):
    predictions_DI = predictions
    return predictions_DI


