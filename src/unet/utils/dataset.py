from typing import Dict, List, Tuple, Callable, Any
import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from PIL import Image
from .parse_label_json import LabelParser
import cv2
import pandas as pd
from transformers import SamProcessor, SamModel
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def contour_process_cv(processed_images, num_masks = 1):
    """
    Processes a dictionary containing image masks and generates statistics.
    
    Parameters:
    processed_images (dict): Dictionary containing image data with masks.
    
    Returns:
    dict: Dictionary with added statistics for each mask.
    """
    for key, result in processed_images.items():
        contours_info = []
        for mask in result['masks']:
            if hasattr(mask, 'numpy'):
                # Convert mask to numpy array if it's not already
                mask = mask.numpy()
            
            if mask.ndim == 4:
                mask = mask.squeeze(0).squeeze(0)
            
            mask = mask.astype(np.uint8)
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Check if there is more than one contour
            if len(contours) > num_masks:
                result['DI'] = None
                break

            for contour in contours:
                convex_hull = cv2.convexHull(contour)
                # Calculate convex hull area
                convex_hull_area = cv2.contourArea(convex_hull)
                # Calculate area
                area = cv2.contourArea(contour)
                # Calculate area ratio between convex hull and contour
                area_ratio = convex_hull_area / area if area != 0 else 0
                
                # Calculate perimeter
                perimeter = cv2.arcLength(convex_hull, True)
                
                # Calculate deformability (perimeter to area ratio) 1-C (2*sqrt(*pi*a))/l
                deformability = 1 - (2*np.sqrt(np.pi*convex_hull_area))/perimeter if convex_hull_area != 0 else 0
                

                contours_info.append({
                    'contour': contour,
                    'area': convex_hull_area,
                    'deformability': deformability,
                    'area_ratio': area_ratio
                })
        
        result['DI'] = contours_info

    return processed_images


def load_partition_dict(partition_dict: Dict[str, Any]) -> Dict[str, Any]:
    logger.info(f"Loading {len(partition_dict)} partitions")
    
    loaded_data = {}
    for key, value in partition_dict.items():
        logger.info(f"Processing partition: {key}")
        
        try:
            if hasattr(value, 'load'):
                loaded_data[key] = value.load()
                logger.info(f"Loaded partition {key} using load() method")
            else:
                loaded_data[key] = value
                logger.info(f"Using direct value for partition {key}")
        except Exception as e:
            logger.error(f"Failed to load partition {key}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(loaded_data)} partitions")
    return loaded_data

# sam_inference

class PrepareSAMDataset(Dataset):
    def __init__(self, images_dict, roi, processor):
        self.images = images_dict
        self.roi = roi.iloc[0].values.astype(float)
        self.processor = processor
        self.keys = sorted(self.images.keys(), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf'))


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        image = self.images[key]()
        # Convert ROI from x, y, width, height to x_min, y_min, x_max, y_max
        x, y, width, height = map(int, self.roi)
        # cropped_image = original_image[y:y+height, x:x+width]
        box_prompt = [x, y, x + width, y + height]
        # Store original image before processing
        image = np.array(image)
        original_image = image.copy()
        cropped_image = original_image[y:y+height, x:x+width]
        image = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image
        

        # Process the image
        inputs = self.processor(
            images=image,
            input_boxes=[[box_prompt]],
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Add original image to inputs
        inputs["cropped_image"] = cropped_image
        inputs["roi"] = self.roi

        return inputs

class SamInference:
    def __init__(self, model_path, processor_path):
        # Load the processor and model
        self.processor = SamProcessor.from_pretrained(processor_path)
        self.model = SamModel.from_pretrained(model_path)
        self.model.eval()  # Set the model to evaluation mode
        # Set model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print("Device: ", next(self.model.parameters()).device)

    def perform_inference(self, dataloader):
        results = {}
        for i, batch in enumerate(dataloader):
            # Get the key from the dataloader
            key = dataloader.dataset.keys[i]  # Using the keys from PrepareSAMDataset
            
            pixel_values = batch['pixel_values'].to(self.device)
            input_boxes = batch['input_boxes'].to(self.device)
            # original_image = batch['original_image']

            # Perform inference
            with torch.no_grad():
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_boxes=input_boxes,
                    multimask_output=False
                )
                predicted_masks = outputs.pred_masks

            # Post-process the predicted masks
            masks = self.processor.image_processor.post_process_masks(
                predicted_masks,
                batch["original_sizes"],
                batch["reshaped_input_sizes"]
            )
            # Convert ROI tensor to integers
            roi = batch['roi'].squeeze().cpu().tolist()  # Convert tensor to list
            # crop masks to roi
            x, y, width, height = map(int, roi)
            # convert masks to numpy
            masks = [mask.cpu().numpy() for mask in masks]
            # squeeze masks
            masks = [np.squeeze(mask).astype(np.uint8) for mask in masks]
            masks = [mask[y:y+height, x:x+width] for mask in masks]
            
            # Store the original image and mask with the original key
            results[key] = {
                'cropped_image': np.array(batch['cropped_image']).squeeze(),
                'masks': masks
            }

        return results
    
class TuneSAMDataset(Dataset):
    def __init__(self, data_dict, processor, prefix=''):
        self.images = data_dict['images']
        self.masks = data_dict['masks']
        self.box_prompt = data_dict['box_prompt']
        self.processor = processor
        self.prefix = prefix
        
        # Standardize and sort keys
        def extract_number(key: str) -> int:
            # Extract numeric part from the key
            nums = ''.join(filter(str.isdigit, key))
            return int(nums) if nums else float('inf')
        
        # Convert keys to list and sort them based on numeric value
        self.keys = list(self.images.keys())
        try:
            self.keys.sort(key=extract_number)
        except Exception as e:
            logger.error(f"Error sorting keys: {e}")
            logger.error(f"Keys sample: {self.keys[:5]}")
            raise
            
        logger.info(f"Dataset initialized with {len(self.keys)} images")
        logger.info(f"Sample keys: {self.keys[:5]}")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer, got {type(idx)}")
            
        if idx >= len(self.keys):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.keys)} items")
            
        key = self.keys[idx]
        logger.debug(f"Accessing item with idx {idx}, key {key}")
        
        image = self.images[key]
        ground_truth_mask = self.masks[key]

        # Store original image before processing
        original_image = image.copy()
        original_ground_truth_mask = ground_truth_mask.copy()

        # Process both image and mask together
        inputs = self.processor(
            images=image,
            segmentation_maps=ground_truth_mask,
            input_boxes=[[self.box_prompt.tolist()]],
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}
        
        # Add metadata to track source
        inputs["dataset_key"] = f"{self.prefix}{key}"
        inputs["ground_truth_mask"] = inputs.pop('labels')
        inputs["original_image"] = original_image
        inputs["original_ground_truth_mask"] = original_ground_truth_mask
        
        return inputs

# opencv_benchmark

def select_background(
    partition: Dict[str, Callable[[], Any]],
    roi: pd.DataFrame
) -> pd.DataFrame:
    """
    Select the background image from the partition. 
    Use left and right arrow keys to navigate through the images.
    Press enter to confirm selection.
    
    Args:
        partition: Kedro partition containing images as load functions
        roi: DataFrame containing ROI coordinates
        
    Returns:
        DataFrame containing background image key
    """
    # Convert partition keys to sorted list for navigation
    keys = sorted(partition.keys())
    current_idx = 0
    
    # Extract ROI coordinates
    x, y, w, h = roi.iloc[0].values.astype(int)
    
    while True:
        # Load and display current image
        current_key = keys[current_idx]
        current_image = partition[current_key]()
        current_image = np.array(current_image)
        
        # Crop to ROI
        cropped_image = current_image[y:y+h, x:x+w]
        
        # Create window with instructions
        cv2.namedWindow('Background Image Selection', cv2.WINDOW_NORMAL)
        info_image = np.zeros((100, cropped_image.shape[1]), dtype=np.uint8)
        cv2.putText(info_image, f"Image: {current_key}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_image, "Left/Right: Navigate, Enter: Select", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Combine info and image
        display_image = np.vstack([info_image, cropped_image])
        cv2.imshow('Background Image Selection', display_image)
        
        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('\r') or key == ord('\n'):  # Enter key
            selected_key = current_key
            break
        elif key == 81 or key == ord('a'):  # Left arrow or 'a'
            current_idx = (current_idx - 1) % len(keys)
        elif key == 83 or key == ord('d'):  # Right arrow or 'd'
            current_idx = (current_idx + 1) % len(keys)
        elif key == 27:  # ESC
            selected_key = keys[0]  # Default to first image
            break
    
    cv2.destroyAllWindows()
    
    # Return selected key in DataFrame
    return pd.DataFrame({'background_key': [selected_key]})

def select_roi(
    partition: Dict[str, Callable[[], Any]]
) -> pd.DataFrame:
    """
    Interactive ROI selection process using OpenCV.
    
    Args:
        partition: Kedro partition containing images as load functions
        
    Returns:
        DataFrame containing ROI coordinates (x, y, width, height)
    """
    # Get the first image for ROI selection
    first_key = next(iter(partition.keys()))
    first_image = partition[first_key]()
    image = np.array(first_image)
    
    # Create window and instructions
    window_name = 'ROI Selection'
    cv2.namedWindow(window_name)
    print("Instructions:")
    print("1. Draw a rectangle by clicking and dragging")
    print("2. Press SPACE or ENTER to confirm selection")
    print("3. Press 'c' to cancel and retry")
    
    # Get ROI using selectROI
    x, y, w, h = cv2.selectROI(window_name, image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    
    # Create DataFrame with ROI coordinates
    roi_df = pd.DataFrame({
        'x': [x],
        'y': [y],
        'width': [w],
        'height': [h]
    })
    
    logger.info(f"Selected ROI: x={x}, y={y}, width={w}, height={h}")
    return roi_df


def filter_empty_frames_ext_bg(
    partition: Dict[str, Callable[[], Any]],
    parameters: Dict[str, Any],
    roi: pd.DataFrame,
    background: pd.DataFrame
) -> Dict[str, Image.Image]:
    """
    Filter out frames that are empty within the ROI using the same processing pipeline as the main processing.
    
    Args:
        partition: Dictionary of image load functions
        parameters: Processing parameters
        roi: DataFrame containing ROI coordinates (x, y, width, height)
        background: DataFrame containing selected background image key
        
    Returns:
        Dictionary of filtered PIL Images (including background)
    """
    logger.info("Filtering empty frames")
    logger.info(f"Initial frame count: {len(partition)}")
    
    # Extract ROI coordinates
    x = roi['x'].iloc[0]
    y = roi['y'].iloc[0]
    w = roi['width'].iloc[0]
    h = roi['height'].iloc[0]
    
    # Setup
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, parameters["kernel_size"])
    filtered_images = {}
    all_areas = []
    
    # Get background image using the selected key from background DataFrame
    bg_key = background['background_key'].iloc[0]
    background_pil = partition[bg_key]()
    background = np.array(background_pil)
    background = background[y:y+h, x:x+w]
    filtered_images[bg_key] = background_pil
    
    # Process each image
    for key, load_func in partition.items():
        if key == bg_key:  # Skip the background image
            continue
            
        # Load and process image
        image_pil = load_func()
        image = np.array(image_pil)
        image = image[y:y+h, x:x+w]
        
        # Apply processing
        blurred_bg = cv2.GaussianBlur(background, parameters["blur_size"], 0)
        blurred = cv2.GaussianBlur(image, parameters["blur_size"], 0)
        bg_sub = cv2.subtract(blurred_bg, blurred)
        _, binary = cv2.threshold(bg_sub, parameters["threshold"], 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        dilate1 = cv2.dilate(binary, kernel, iterations=1)
        erode1 = cv2.erode(dilate1, kernel, iterations=1)
        erode2 = cv2.erode(erode1, kernel, iterations=1)
        processed = cv2.dilate(erode2, kernel, iterations=1)
        
        # Calculate area
        white_area = np.sum(processed > 0)
        all_areas.append(white_area)
        
        logger.debug(f"Frame {key}: white area in ROI = {white_area}")
        
        # Keep frame if it has sufficient white pixels in ROI
        if white_area > parameters["minimum_area"]:
            filtered_images[key] = image_pil
            logger.debug(f"Keeping frame {key}")
        else:
            logger.debug(f"Filtering out frame {key}")
    
    # Log statistics
    if all_areas:
        logger.info(f"Area statistics within ROI:")
        logger.info(f"Min area: {min(all_areas)}")
        logger.info(f"Max area: {max(all_areas)}")
        logger.info(f"Mean area: {np.mean(all_areas):.2f}")
        logger.info(f"Median area: {np.median(all_areas):.2f}")
    
    logger.info(f"Filtered {len(partition) - len(filtered_images)} empty frames")
    logger.info(f"Remaining frames: {len(filtered_images)}")
    
    return filtered_images


def filter_empty_frames(
    partition: Dict[str, Callable[[], Any]],
    parameters: Dict[str, Any],
    roi: pd.DataFrame
) -> Dict[str, Image.Image]:
    """
    Filter out frames that are empty within the ROI using the same processing pipeline as the main processing.
    
    Args:
        partition: Dictionary of image load functions
        parameters: Processing parameters
        roi: DataFrame containing ROI coordinates (x, y, width, height)
        
    Returns:
        Dictionary of filtered PIL Images (including background)
    """
    logger.info("Filtering empty frames")
    logger.info(f"Initial frame count: {len(partition)}")
    
    # Extract ROI coordinates
    x = roi['x'].iloc[0]
    y = roi['y'].iloc[0]
    w = roi['width'].iloc[0]
    h = roi['height'].iloc[0]
    
    # Setup
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, parameters["kernel_size"])
    filtered_images = {}
    all_areas = []
    
    # Get background image and store as PIL Image
    bg_key = next(key for key in partition.keys() if "background" in key.lower())
    background_pil = partition[bg_key]()
    background = np.array(background_pil)
    background = background[y:y+h, x:x+w]
    filtered_images[bg_key] = background_pil
    
    # Process each image
    for key, load_func in partition.items():
        if "background" in key.lower():
            continue
            
        # Load and process image
        image_pil = load_func()
        image = np.array(image_pil)
        image = image[y:y+h, x:x+w]
        
        # Apply processing
        blurred_bg = cv2.GaussianBlur(background, parameters["blur_size"], 0)
        blurred = cv2.GaussianBlur(image, parameters["blur_size"], 0)
        bg_sub = cv2.subtract(blurred_bg, blurred)
        _, binary = cv2.threshold(bg_sub, parameters["threshold"], 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        dilate1 = cv2.dilate(binary, kernel, iterations=1)
        erode1 = cv2.erode(dilate1, kernel, iterations=1)
        erode2 = cv2.erode(erode1, kernel, iterations=1)
        processed = cv2.dilate(erode2, kernel, iterations=1)
        
        # Calculate area
        white_area = np.sum(processed > 0)
        all_areas.append(white_area)
        
        logger.debug(f"Frame {key}: white area in ROI = {white_area}")
        
        # Keep frame if it has sufficient white pixels in ROI
        if white_area > parameters["minimum_area"]:
            filtered_images[key] = image_pil
            logger.debug(f"Keeping frame {key}")
        else:
            logger.debug(f"Filtering out frame {key}")
    
    # Log statistics
    if all_areas:
        logger.info(f"Area statistics within ROI:")
        logger.info(f"Min area: {min(all_areas)}")
        logger.info(f"Max area: {max(all_areas)}")
        logger.info(f"Mean area: {np.mean(all_areas):.2f}")
        logger.info(f"Median area: {np.median(all_areas):.2f}")
    
    logger.info(f"Filtered {len(partition) - len(filtered_images)} empty frames")
    logger.info(f"Remaining frames: {len(filtered_images)}")
    
    return filtered_images

def _validate_image_shapes(
    images: Dict[str, Callable[[], Any]], 
    condition: str,
    sample_size: int = 5
) -> Tuple[Dict[str, np.ndarray], tuple]:
    """
    Helper function to load images and validate their shapes.
    
    Args:
        images: Dictionary of image load functions
        condition: Name of the condition for logging (e.g., 'in_focus')
        sample_size: Number of images to check for shape consistency
        
    Returns:
        Tuple containing:
        - Dictionary of loaded images as numpy arrays
        - Reference shape of the images
        
    Raises:
        ValueError: If inconsistent shapes are detected
    """
    loaded_images = {}
    reference_shape = None
    
    for i, (key, load_func) in enumerate(images.items()):
        img = np.array(load_func())
        loaded_images[key] = img
        
        # Shape validation for first n images
        if i < sample_size:
            current_shape = img.shape
            if reference_shape is None:
                reference_shape = current_shape
                logger.info(f"{condition} reference shape: {current_shape}")
            elif current_shape != reference_shape:
                raise ValueError(
                    f"Inconsistent shape in {condition} images. "
                    f"Expected {reference_shape}, "
                    f"got {current_shape} for image {key}"
                )
    
    return loaded_images, reference_shape
    
def _standardize_key(key: str) -> str:
    """Standardize key to XXXX format"""
    # Remove 'image.' prefix if present
    if key.startswith('image.'):
        key = key.replace('image.', '')
    # Ensure 4-digit padding
    return key.zfill(4)

# opencv_benchmark

class SegmentationDataset(Dataset):
    def __init__(self, images: Dict, labels: List[Dict]):
        self.images = images
        self.masks = LabelParser.parse_json(labels)
        logger.debug(f"self.images type: {type(self.images)}")
        logger.debug(f"self.images class name: {self.images.__class__.__name__}")
        
        # Extract numbers from keys and create mapping
        def get_number(key: str) -> str:
            # For image.XXXX format
            if key.startswith('image.'):
                return key.split('.')[1]
            # For XXXX.tiff format
            return key.split('.')[0]
        
        # Create mappings from numbers to full keys
        image_number_map = {get_number(k): k for k in self.images.keys()}
        mask_number_map = {get_number(k): k for k in self.masks.keys()}
        
        # Find matching numbers
        image_numbers = set(image_number_map.keys())
        mask_numbers = set(mask_number_map.keys())
        common_numbers = image_numbers.intersection(mask_numbers)
        
        logger.info(f"Number of image keys: {len(self.images)}")
        logger.info(f"Number of mask keys: {len(self.masks)}")
        logger.info(f"Number of matching numbers: {len(common_numbers)}")
        
        # Store matching pairs
        self.image_mask_pairs = [(image_number_map[num], mask_number_map[num]) 
                                for num in common_numbers]
        
        # Log some sample matches
        if self.image_mask_pairs:
            logger.debug("Sample matches (image key -> mask key):")
            for img_key, mask_key in self.image_mask_pairs[:5]:
                logger.debug(f"  {img_key} -> {mask_key}")
        
        # Log unmatched numbers
        unmatched_images = image_numbers - mask_numbers
        unmatched_masks = mask_numbers - image_numbers
        if unmatched_images:
            logger.warning(f"Numbers only in images ({len(unmatched_images)}): {list(unmatched_images)[:5]}")
        if unmatched_masks:
            logger.warning(f"Numbers only in masks ({len(unmatched_masks)}): {list(unmatched_masks)[:5]}")
        
        if not self.image_mask_pairs:
            logger.error("No matching pairs found!")

    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        # Get image and mask keys from the pairs
        image_key, mask_key = self.image_mask_pairs[idx]
        
        # Get raw data
        image = self.images[image_key]
        mask = self.masks[mask_key]
    
        # Handle image data
        if callable(image):
            logger.debug(f"Calling method to get image data for {image_key}")
            image = image()
    
        # Convert to numpy arrays with explicit dtype
        try:
            image = np.asarray(image, dtype=np.float32)
            mask = np.asarray(mask, dtype=np.float32)
            
            logger.debug(f"Image shape: {image.shape}, dtype: {image.dtype}")
            logger.debug(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        except Exception as e:
            logger.error(f"Error converting arrays for key {image_key}: {str(e)}")
            raise
        
        # Convert to torch tensors
        image = torch.FloatTensor(image)
        mask = torch.FloatTensor(mask)
        
        return image, mask

# cv_processing

class PrepareOpenCVDataset(Dataset):
    def __init__(self, images_dict, roi, config_json):
        self.images = images_dict
        # Convert ROI to same format as SAM pipeline
        self.roi = roi.iloc[0].values.astype(float)  # [x, y, width, height]
        self.config = config_json
        # Sort keys similar to SAM dataset
        self.keys = sorted(self.images.keys(), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else float('inf'))
        
        # Initialize background image
        self.background = self._init_background()
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS, 
            self._ensure_tuple(config_json["morph_kernel_size"])
        )

    def _ensure_tuple(self, value):
        if isinstance(value, int):
            return (value, value)
        return tuple(value)

    def _init_background(self):
        # Find background image
        bg_keys = [key for key in self.images.keys() if "background_clean" in key.lower()]
        if not bg_keys:
            raise ValueError("No background image found in dataset")
        bg_key = bg_keys[0]

        # Load the background image
        background = self.images[bg_key]()
        if not isinstance(background, Image.Image):
            raise ValueError(f"Background image {bg_key} is not a valid PIL image")

        background = np.array(background)
        if background.ndim == 0:
            raise ValueError(f"Background image {bg_key} could not be converted to a valid NumPy array")

        # Crop background to ROI using unpacked values
        x, y, w, h = self.roi
        return background[int(y):int(y+h), int(x):int(x+w)]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        if "background_clean" in key.lower():
            return None

        # Load and process image
        img = self.images[key]()
        if not isinstance(img, Image.Image):
            logger.warning(f"Image {key} is not a valid PIL image")
            return None

        # Convert to numpy and crop to ROI using unpacked values
        x, y, w, h = self.roi
        image = np.array(img)
        image = image[int(y):int(y+h), int(x):int(x+w)]

        # Apply processing steps
        blurred_bg = cv2.GaussianBlur(
            self.background, 
            self._ensure_tuple(self.config["gaussian_blur_size"]), 
            0
        )
        blurred_targ = cv2.GaussianBlur(
            image, 
            self._ensure_tuple(self.config["gaussian_blur_size"]), 
            0
        )
        bg_sub = cv2.subtract(blurred_bg, blurred_targ)
        _, binary = cv2.threshold(
            bg_sub, 
            self.config["bg_subtract_threshold"], 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Morphological operations
        dilate1 = cv2.dilate(binary, self.kernel, iterations=1)
        erode1 = cv2.erode(dilate1, self.kernel, iterations=1)
        erode2 = cv2.erode(erode1, self.kernel, iterations=1)
        processed = cv2.dilate(erode2, self.kernel, iterations=1)
        
        processed = processed.astype(np.uint8)
        
        # Return in format similar to SAM output
        return {
            'cropped_image': image,
            'masks': [processed],  # Wrap in list to match SAM format
        }

class OpenCVInference:
    def __init__(self, config_json):
        self.config = config_json

    def perform_processing(self, dataset):
        results = {}
        for i in range(len(dataset)):
            batch = dataset[i]
            if batch is None:
                continue
                
            key = dataset.keys[i]
            results[key] = {
                'cropped_image': batch['cropped_image'],
                'masks': batch['masks'],
            }
        return results
            