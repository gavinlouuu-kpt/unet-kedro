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
from scipy.stats import gaussian_kde
import torch  # Assuming PyTorch tensors, adjust if using a different library

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

def create_scatter_plots_with_csv(collection: Dict[str, Any]) -> Tuple[Dict[str, Image.Image], Dict[str, pd.DataFrame], Image.Image]:
    """
    Creates individual scatter plots and corresponding CSV files for each PKL file,
    and a combined scatter plot integrating all data.
    
    Args:
        collection: Dictionary containing the results
        
    Returns:
        Tuple containing:
        - Dictionary mapping keys to PIL Images of scatter plots
        - Dictionary mapping keys to pandas DataFrames for CSV files
        - PIL Image of combined scatter plot
    """
    scatter_plots = {}
    csv_data = {}
    all_areas = []
    all_deformabilities = []
    
    # Process each PKL file
    for pkl_name, value in collection.items():
        # logger.info(f"Processing PKL: {pkl_name}")
        
        # Load data on-demand
        if callable(value):
            try:
                result = value()
                # logger.info(f"Successfully loaded data for {pkl_name}")
            except Exception as e:
                # logger.error(f"Error loading data for {pkl_name}: {e}")
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
                        all_areas.append(contour_info['area'])
                        all_deformabilities.append(contour_info['deformability'])
        
        # Calculate point density for each PKL file individually
        if areas and deformabilities:
            xy = np.vstack([areas, deformabilities])
            z = gaussian_kde(xy)(xy)
            
            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            x, y, z = np.array(areas)[idx], np.array(deformabilities)[idx], z[idx]
            
            # Plot points with density-based coloring
            plt.scatter(x, y, c=z, s=50, edgecolor='face', cmap='viridis')
            
            plt.xlabel('Area')
            plt.ylabel('Deformability')
            plt.ylim(0, 1)
            title = f'Area vs Deformability Density - {pkl_name}'
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
            # logger.info(f"Created density plot for {image_key} with {len(areas)} points")
            
            # Create DataFrame and save to CSV without index
            df = pd.DataFrame({
                'Area': areas, 
                'Deformability': deformabilities
            }, index=None)
            
            csv_key = f"{pkl_name}"
            csv_data[csv_key] = df
            # logger.info(f"Created CSV data for {csv_key}")
        else:
            logger.warning(f"No data points to plot for {pkl_name}")
        
        plt.close(fig)  # Close the figure to free memory
    
    # Create combined scatter plot
    if all_areas and all_deformabilities:
        fig_combined = plt.figure(figsize=(12, 10))
        xy_combined = np.vstack([all_areas, all_deformabilities])
        z_combined = gaussian_kde(xy_combined)(xy_combined)
        
        idx_combined = z_combined.argsort()
        x_combined, y_combined, z_combined = np.array(all_areas)[idx_combined], np.array(all_deformabilities)[idx_combined], z_combined[idx_combined]
        
        plt.scatter(x_combined, y_combined, c=z_combined, s=50, edgecolor='face', cmap='viridis')
        plt.xlabel('Area')
        plt.ylabel('Deformability')
        plt.title('Combined Area vs Deformability Density Plot')
        plt.grid(True, alpha=0.3)
        
        buf_combined = BytesIO()
        plt.savefig(buf_combined, format='png', bbox_inches='tight', dpi=300)
        buf_combined.seek(0)
        combined_image = Image.open(buf_combined).copy()
        buf_combined.close()
        plt.close(fig_combined)
        
        logger.info("Successfully created combined density plot")
    else:
        combined_image = None
        logger.warning("No data points to create combined plot")
    
    return scatter_plots, csv_data, combined_image

def create_combined_scatter_plot(collection: Dict[str, Any]) -> Tuple[Image.Image, pd.DataFrame]:
    """
    Creates a single scatter plot combining data from all PKL files with different colors
    
    Args:
        collection: Dictionary containing the results
        
    Returns:
        Tuple containing:
        - PIL Image of combined scatter plot
        - Consolidated pandas DataFrame with source information
    """
    fig = plt.figure(figsize=(12, 10))
    all_data = []
    
    # Process each PKL file
    for pkl_name, value in collection.items():
        logger.info(f"Processing PKL for combined plot: {pkl_name}")
        
        # Load data on-demand
        if callable(value):
            try:
                result = value()
            except Exception as e:
                logger.error(f"Error loading data for {pkl_name}: {e}")
                continue
        else:
            result = value
        
        areas = []
        deformabilities = []
        
        # Collect data points
        for image_data in result.values():
            if 'DI' in image_data and image_data['DI']:
                for contour_info in image_data['DI']:
                    if isinstance(contour_info, dict) and 'area' in contour_info and 'deformability' in contour_info:
                        areas.append(contour_info['area'])
                        deformabilities.append(contour_info['deformability'])
                        all_data.append({
                            'Area': contour_info['area'],
                            'Deformability': contour_info['deformability'],
                            'Source': pkl_name
                        })
        
        if areas and deformabilities:
            # Calculate and plot density for each PKL file individually
            xy = np.vstack([areas, deformabilities])
            z = gaussian_kde(xy)(xy)
            
            # Sort points by density
            idx = z.argsort()
            x, y, z = np.array(areas)[idx], np.array(deformabilities)[idx], z[idx]
            
            # Plot with unique label for legend
            scatter = plt.scatter(x, y, c=z, label=pkl_name, alpha=0.6, s=50)
            
            logger.info(f"Added {len(areas)} points from {pkl_name} to combined plot")
    
    plt.xlabel('Area')
    plt.ylabel('Deformability')
    plt.ylim(0, 1)
    plt.title('Combined Area vs Deformability Density Plot')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    combined_image = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    
    # Create consolidated DataFrame
    combined_df = pd.DataFrame(all_data)
    
    logger.info("Successfully created combined density plot")
    return combined_image, combined_df

from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import (
    ColumnDataSource, 
    HoverTool, 
    ColorBar, 
    LinearColorMapper,
    BasicTicker
)
from bokeh.embed import file_html
from bokeh.resources import CDN
import numpy as np
from scipy.stats import gaussian_kde
import base64
import cv2

def create_interactive_scatter_plots(collection: Dict[str, Any]) -> str:
    """Creates an interactive scatter plot using Bokeh"""
    # Prepare data
    data = {
        'area': [],
        'deformability': [],
        'source': [],
        'image': [],
    }
    
    # Process each PKL file
    for pkl_name, value in collection.items():
        logger.info(f"Processing PKL: {pkl_name}")
        result = value() if callable(value) else value
        
        for image_key, image_data in result.items():
            if 'DI' in image_data and image_data['DI']:
                # Convert image to base64
                img_array = image_data['cropped_image']
                mask = image_data['masks'][0] if image_data.get('masks') else None
                
                # Convert grayscale to RGB
                if len(img_array.shape) == 2:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                
                # Create mask overlay
                if mask is not None:
                    # Create red overlay with same shape as RGB image
                    overlay = np.zeros_like(img_array)
                    overlay[mask > 0] = [255, 0, 0]  # Red color
                    # Blend original and overlay
                    img_array = cv2.addWeighted(img_array, 0.8, overlay, 0.2, 0)
                
                # Convert to base64
                img_pil = Image.fromarray(img_array)
                buffered = BytesIO()
                img_pil.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                for contour_info in image_data['DI']:
                    if isinstance(contour_info, dict) and 'area' in contour_info and 'deformability' in contour_info:
                        data['area'].append(contour_info['area'])
                        data['deformability'].append(contour_info['deformability'])
                        data['source'].append(pkl_name)
                        data['image'].append(img_str)

    # Calculate point density
    xy = np.vstack([data['area'], data['deformability']])
    density = gaussian_kde(xy)(xy)
    data['density'] = density.tolist()

    # Create ColumnDataSource
    source = ColumnDataSource(data)

    # Create color mapper
    color_mapper = LinearColorMapper(
        palette="Viridis256",
        low=min(density),
        high=max(density)
    )

    # Create figure
    p = figure(
        # width=800, 
        # height=600,
        sizing_mode="stretch_both",
        title="Interactive Area vs Deformability Plot",
        tools="pan,box_zoom,reset,save,wheel_zoom",
        active_scroll="wheel_zoom"
    )

    # Add scatter points
    scatter = p.scatter(
        'area',
        'deformability',
        source=source,
        size=10,
        fill_color={'field': 'density', 'transform': color_mapper},
        line_color=None,
        alpha=0.6
    )

    # Add color bar
    color_bar = ColorBar(
        color_mapper=color_mapper,
        ticker=BasicTicker(),
        label_standoff=12,
        border_line_color=None,
        location=(0,0),
        title="Density",
        orientation='vertical'
    )
    p.add_layout(color_bar, 'right')

    # Configure axis labels
    p.xaxis.axis_label = 'Area'
    p.yaxis.axis_label = 'Deformability'

    # Add hover tool
    hover = HoverTool(
        tooltips="""
        <div style="background-color: white; opacity: 1; padding: 10px;">
            <div>
                <span style="font-size: 14px;"><b>Source:</b> @source</span>
            </div>
            <div>
                <span style="font-size: 14px;"><b>Area:</b> @area{0.00}</span>
            </div>
            <div>
                <span style="font-size: 14px;"><b>Deformability:</b> @deformability{0.00}</span>
            </div>
            <div>
                <img src="data:image/png;base64,@image" style="width: 200px;">
            </div>
        </div>
        """
    )
    p.add_tools(hover)

    # Generate HTML
    html = file_html(p, CDN, "Interactive Scatter Plot")
    return html