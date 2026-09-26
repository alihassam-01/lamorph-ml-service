from typing import Any, Dict
import numpy as np
from PIL import Image

def prepare_eyebag_input(image: Image.Image, box: Dict[str, int]) -> np.ndarray:
    """
    Given a full image and a bounding box, crop the eye region and format it
    to NCHW float32 [1, 3, 160, 256].
    """
    # Crop
    crop = image.crop((
        box['x'],
        box['y'],
        box['x'] + box['width'],
        box['y'] + box['height']
    ))
    # Resize to 256x160
    crop = crop.resize((256, 160), Image.Resampling.BILINEAR)

    # Convert to numpy
    arr = np.array(crop).astype(np.float32) / 255.0

    # Normalize (ImageNet)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # HWC to CHW
    arr = np.transpose(arr, (2, 0, 1))

    # Add batch dim
    arr = np.expand_dims(arr, axis=0)

    return arr
