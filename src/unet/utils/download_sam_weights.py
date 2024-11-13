import os
import urllib.request
import logging
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAM_CHECKPOINTS = {
    "h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth"
    },
    "l": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "filename": "sam_vit_l_0b3195.pth"
    },
    "b": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "filename": "sam_vit_b_01ec64.pth"
    }
}

def download_sam_weights() -> None:
    # Get project root directory (3 levels up from this script)
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Define base save directory
    save_dir = project_root / "data" / "06_models" / "sam_weights"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for variant, info in SAM_CHECKPOINTS.items():
        save_path = save_dir / info["filename"]
        url = info["url"]
        
        # Check if file already exists
        if save_path.exists():
            logger.info(f"SAM {variant.upper()} weights already exist at {save_path}")
            continue
        
        # Download weights
        logger.info(f"Downloading SAM {variant.upper()} weights from {url}")
        logger.info(f"This might take a while...")
        
        try:
            urllib.request.urlretrieve(url, str(save_path))
            logger.info(f"Successfully downloaded SAM {variant.upper()} weights to {save_path}")
        except Exception as e:
            logger.error(f"Error downloading SAM {variant.upper()} weights: {str(e)}")
            raise

if __name__ == "__main__":
    download_sam_weights()