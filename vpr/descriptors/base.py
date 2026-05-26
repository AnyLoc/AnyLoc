"""Shared helpers for the codebook export pipeline."""
from __future__ import annotations

import cv2
import numpy as np


def load_gray(path) -> np.ndarray:
    """Load an image as an HxW uint8 grayscale array."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img
