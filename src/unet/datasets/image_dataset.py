from pathlib import PurePosixPath, Path
from typing import Any, Dict
import numpy as np
from PIL import Image
import fsspec


from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path

from typing import Dict, Any, List, Tuple
import torch
from torch.utils.data import Dataset

class ImageDataset(AbstractDataset[np.ndarray, np.ndarray]):
    """``ImageDataset`` loads / save image data from a given filepath as `numpy` array using Pillow.

    Example:
    ::

        >>> ImageDataset(filepath='/img/file/path.png')
    """

    def __init__(self, filepath: str):
        """Creates a new instance of ImageDataset to load / save image data for given filepath.

        Args:
            filepath: The location of the image file to load / save data.
        """
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)

    def load(self) -> np.ndarray:
        """Loads data from the image file.

        Returns:
            Data from the image file as a numpy array
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path, mode="rb") as f:  # Open in binary mode
            image = Image.open(f).convert("RGBA")
            return np.asarray(image)

    def save(self, data: np.ndarray) -> None:
        """Saves image data to the specified filepath."""
        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, mode="wb") as f:  # Open in binary mode
            image = Image.fromarray(data)
            image.save(f)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(filepath=self._filepath, protocol=self._protocol)