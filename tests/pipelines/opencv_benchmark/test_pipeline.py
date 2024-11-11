"""
Test pipeline for opencv_benchmark
"""
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

from unet.pipelines.opencv_benchmark.nodes import (
    load_cv_processed_images,
    load_label_masks
)
from unet.pipelines.opencv_benchmark.pipeline import create_pipeline

# Fixtures
@pytest.fixture
def dummy_images():
    """Create consistent test images"""
    def create_load_func():
        return lambda: Image.fromarray(np.zeros((100, 100), dtype=np.uint8))
    
    return {
        f"image_{i}": create_load_func()
        for i in range(10)
    }

@pytest.fixture
def dummy_inconsistent_images():
    """Create images with inconsistent shapes"""
    return {
        "image_1": lambda: Image.fromarray(np.zeros((100, 100))),
        "image_2": lambda: Image.fromarray(np.zeros((100, 100))),
        "image_3": lambda: Image.fromarray(np.zeros((200, 200)))
    }

# Node tests
def test_load_label_masks(dummy_images):
    """Test loading label masks with consistent shapes"""
    # Act
    result = load_label_masks(dummy_images)
    
    # Assert
    assert len(result) == len(dummy_images)
    first_mask = np.array(next(iter(result.values())))
    assert first_mask.shape == (100, 100)

def test_load_label_masks_inconsistent_shapes(dummy_inconsistent_images):
    """Test loading label masks fails with inconsistent shapes"""
    # Assert
    with pytest.raises(ValueError, match="Inconsistent.*shape"):
        load_label_masks(dummy_inconsistent_images)

def test_load_cv_processed_images(dummy_images):
    """Test loading CV processed images with consistent shapes"""
    # Act
    result = load_cv_processed_images(
        cv_processed_in_focus=dummy_images,
        cv_processed_slight_under=dummy_images
    )
    
    # Assert
    assert "in_focus" in result
    assert "slight_under" in result
    assert len(result["in_focus"]) == len(dummy_images)
    assert len(result["slight_under"]) == len(dummy_images)

def test_load_cv_processed_images_shape_mismatch(dummy_images, dummy_inconsistent_images):
    """Test loading CV processed images fails with shape mismatch"""
    # Assert
    with pytest.raises(ValueError, match="Inconsistent.*shape"):
        load_cv_processed_images(
            cv_processed_in_focus=dummy_images,
            cv_processed_slight_under=dummy_inconsistent_images
        )

# Pipeline tests
def test_opencv_benchmark_pipeline(dummy_images):
    """Test the complete opencv benchmark pipeline"""
    # Arrange
    pipeline = create_pipeline()
    
    # Slice the pipeline to test specific nodes
    pipeline = (
        pipeline
        .from_nodes("load_cv_processed_images")
        .to_nodes("load_label_masks")
    )
    
    # Create a catalog with test data
    catalog = DataCatalog({
        "cv_processed_in_focus": MemoryDataSet(data=dummy_images),
        "cv_processed_slight_under": MemoryDataSet(data=dummy_images),
        "label_masks": MemoryDataSet(data=dummy_images)
    })
    
    # Act
    pipeline.run(catalog)
    
    # Assert outputs exist in catalog
    assert catalog.exists("cv_processed_images")
    assert catalog.exists("loaded_masks")
