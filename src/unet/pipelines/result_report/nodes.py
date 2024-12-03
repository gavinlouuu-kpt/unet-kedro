# TODO:
# - Create a node to load the collections of pkl dict
# - Create a side by side image of the original image beside mask overlayed on the original image
# - Save the side by side image to the reporting folder
# - Construct a scatter plot of the (area,deformability) under the key [DI], the scatter plot the dots on the plot should identify the which pkl it is comming from

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from matplotlib.figure import Figure
from io import BytesIO
from PIL import Image
import logging
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class OverlayFigure:
    figure: Figure
    title: str

@dataclass
class ScatterFigure:
    figure: Figure
    title: str
    dataset: str

@dataclass
class ContourMetrics:
    area: float
    deformability: float

@dataclass
class PklMetrics:
    name: str
    metrics: List[ContourMetrics]


def create_mask_overlays(collection: Dict[str, Any]) -> List[OverlayFigure]:
    """
    Creates side-by-side comparisons of original images and mask overlays
    
    Args:
        collection: Dictionary containing the partitioned data
        method_name: Name of the method (CV or SAM)
    
    Returns:
        List of OverlayFigure objects containing the figures and metadata
    """
    overlay_figures = []
    
    # First, load the actual data from the callable
    loaded_data = {}
    for key, value in collection.items():
        logger.info(f"Loading data for {key}")
        if callable(value):
            try:
                loaded_data[key] = value()
                logger.info(f"Successfully loaded data for {key}")
            except Exception as e:
                logger.error(f"Error loading data for {key}: {e}")
                continue
        else:
            loaded_data[key] = value
            logger.info(f"Using direct value for {key}")
    
    # Now process the loaded data
    for dataset_name, dataset_results in loaded_data.items():
        logger.info(f"Processing dataset: {dataset_name}")
        
        if not isinstance(dataset_results, dict):
            logger.warning(f"Skipping {dataset_name}: not a dictionary")
            continue
            
        for img_key, result in dataset_results.items():
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original image
            original = result['original_image']
            ax1.imshow(original, cmap='gray')
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Mask overlay
            ax2.imshow(original, cmap='gray')
            if result['masks']:
                mask = result['masks'][0]
                # Create red overlay
                overlay = np.zeros((*original.shape, 3))
                overlay[mask > 0] = [1, 0, 0]  # Red color
                ax2.imshow(overlay, alpha=0.3)
            ax2.set_title('Mask Overlay')
            ax2.axis('off')
            
            title = f'{dataset_name} - {img_key}'
            plt.suptitle(title)
            
            overlay_figures.append(OverlayFigure(figure=fig, title=title))
            # Close the figure to free memory, the figure is still stored in OverlayFigure
            plt.close(fig)
    
    logger.info(f"Created {len(overlay_figures)} overlay figures")
    return overlay_figures

def create_scatter_plots_with_csv(collection: Dict[str, Any]) -> Tuple[Dict[str, Image.Image], Dict[str, pd.DataFrame]]:
    """
    Creates individual scatter plots and corresponding CSV files for each PKL file
    
    Args:
        collection: Dictionary containing the results
        
    Returns:
        Tuple containing:
        - Dictionary mapping keys to PIL Images of scatter plots
        - Dictionary mapping keys to pandas DataFrames for CSV files
    """
    scatter_plots = {}
    csv_data = {}
    
    # Process each PKL file
    for pkl_name, value in collection.items():
        logger.info(f"Processing PKL: {pkl_name}")
        
        # Load data on-demand
        if callable(value):
            try:
                result = value()
                logger.info(f"Successfully loaded data for {pkl_name}")
            except Exception as e:
                logger.error(f"Error loading data for {pkl_name}: {e}")
                continue
        else:
            result = value
        
        fig = plt.figure(figsize=(10, 8))
        areas = []
        deformabilities = []
        
        # Iterate through all images in the batch
        for image_key, image_data in result.items():
            # Check if DI exists and has data
            if 'DI' in image_data and image_data['DI']:
                for contour_info in image_data['DI']:
                    if isinstance(contour_info, dict) and 'area' in contour_info and 'deformability' in contour_info:
                        areas.append(contour_info['area'])
                        deformabilities.append(contour_info['deformability'])
        
        # Plot points if we have data
        if areas and deformabilities:
            plt.scatter(areas, deformabilities, alpha=0.6)
            
            plt.xlabel('Area')
            plt.ylabel('Deformability')
            title = f'Area vs Deformability - {pkl_name}'
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            # Convert matplotlib figure to PIL Image
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
            buf.seek(0)
            image = Image.open(buf).copy()
            buf.close()
            
            # Save to dictionary with a proper key
            image_key = f"{pkl_name}"
            scatter_plots[image_key] = image
            logger.info(f"Created scatter plot for {image_key} with {len(areas)} points")
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame({'Area': areas, 'Deformability': deformabilities})
            csv_key = f"{pkl_name}"
            csv_data[csv_key] = df
            logger.info(f"Created CSV data for {csv_key}")
        else:
            logger.warning(f"No data points to plot for {pkl_name}")
        
        plt.close(fig)  # Close the figure to free memory
    
    logger.info(f"Successfully created {len(scatter_plots)} scatter plots and CSV files")
    return scatter_plots, csv_data


# def generate_method_report(overlay_paths: Dict[str, str], 
#                          scatter_paths: Dict[str, str], 
#                          params: Dict[str, Any],
#                          method_name: str) -> str:
#     """
#     Generates an HTML report for a single method
#     """
#     output_dir = Path(params["reporting_dir"]) / method_name
#     report_path = output_dir / "report.html"
    
#     html_content = [
#         "<!DOCTYPE html>",
#         "<html>",
#         "<head>",
#         f"<title>{method_name} Analysis Report</title>",
#         "<style>",
#         "body { font-family: Arial, sans-serif; margin: 20px; }",
#         ".image-container { margin: 20px 0; }",
#         "img { max-width: 100%; }",
#         "</style>",
#         "</head>",
#         "<body>",
#         f"<h1>{method_name} Analysis Report</h1>",
        
#         "<h2>Mask Overlays</h2>"
#     ]
    
#     # Add mask overlays
#     for name, path in overlay_paths.items():
#         html_content.extend([
#             f"<div class='image-container'>",
#             f"<h3>{name}</h3>",
#             f"<img src='{os.path.relpath(path, output_dir)}'>",
#             "</div>"
#         ])
    
#     # Add scatter plots
#     html_content.append("<h2>Scatter Plots</h2>")
#     for name, path in scatter_paths.items():
#         html_content.extend([
#             f"<div class='image-container'>",
#             f"<h3>{name}</h3>",
#             f"<img src='{os.path.relpath(path, output_dir)}'>",
#             "</div>"
#         ])
    
#     html_content.extend([
#         "</body>",
#         "</html>"
#     ])
    
#     # Write HTML file
#     report_path.write_text("\n".join(html_content))
    
#     return str(report_path)

