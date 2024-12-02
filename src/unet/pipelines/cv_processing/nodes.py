"""
This is a boilerplate pipeline 'cv_processing'
generated using Kedro 0.19.9
"""

import cv2
import numpy as np
from typing import Dict, Any, Callable
from PIL import Image

def img_process(images, parameters, roi):
    x = roi['x'].iloc[0]
    y = roi['y'].iloc[0]
    w = roi['width'].iloc[0]
    h = roi['height'].iloc[0]

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, parameters["kernel_size"])
    processed_images = {}       

    return