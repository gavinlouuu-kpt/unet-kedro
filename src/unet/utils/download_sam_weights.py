import os
import urllib.request
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_sam_weights():
    # Get project root directory (3 levels up from this script)
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Define paths
    save_dir = project_root / "data" / "06_models"
    filename = "sam_vit_h_4b8939.pth"
    save_path = save_dir / filename
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    
    # Create directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists
    if save_path.exists():
        logger.info(f"SAM weights already exist at {save_path}")
        return
    
    # Download weights
    logger.info(f"Downloading SAM weights from {url}")
    logger.info(f"This might take a while... (file size ~2.4GB)")
    
    try:
        urllib.request.urlretrieve(url, str(save_path))
        logger.info(f"Successfully downloaded SAM weights to {save_path}")
    except Exception as e:
        logger.error(f"Error downloading SAM weights: {str(e)}")
        raise

if __name__ == "__main__":
    download_sam_weights()