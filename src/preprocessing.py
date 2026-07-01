from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def load_image(image_or_path: str | Path | np.ndarray) -> np.ndarray:
    """Return an RGB uint8 image array."""
    if isinstance(image_or_path, np.ndarray):
        image = image_or_path
    else:
        path = Path(image_or_path)
        if not path.exists():
            raise FileNotFoundError(path)
        image = np.asarray(Image.open(path).convert("RGB"))

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB image shape, got {image.shape}")
    return image.astype(np.uint8, copy=False)


def crop_black_borders(image_or_path: str | Path | np.ndarray, threshold: int = 7, padding: int = 8) -> np.ndarray:
    image = load_image(image_or_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if not mask.any():
        return image

    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    y0 = max(0, y0 - padding)
    x0 = max(0, x0 - padding)
    y1 = min(image.shape[0], y1 + padding)
    x1 = min(image.shape[1], x1 + padding)
    return image[y0:y1, x0:x1]


def resize_image(image_or_path: str | Path | np.ndarray, size: int | Iterable[int] = 224) -> np.ndarray:
    image = load_image(image_or_path)
    if isinstance(size, int):
        width = height = size
    else:
        width, height = size
    return cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def preprocess_image(image_or_path: str | Path | np.ndarray, size: int = 224) -> np.ndarray:
    return resize_image(crop_black_borders(image_or_path), size=size)


def normalize_image(image_or_path: str | Path | np.ndarray) -> np.ndarray:
    return load_image(image_or_path).astype(np.float32) / 255.0


def extract_handcrafted_features(image_or_path: str | Path | np.ndarray, size: int = 224) -> np.ndarray:
    """Small baseline feature vector for sklearn models.

    This is not a replacement for the final CNN models; it makes Week 1 executable
    before TensorFlow/PyTorch and dataset weights exist.
    """
    image = preprocess_image(image_or_path, size=size)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 50, 150)

    features: list[float] = []
    for arr in (image, hsv):
        channels = cv2.split(arr)
        for channel in channels:
            features.extend([float(channel.mean()), float(channel.std())])
            hist = cv2.calcHist([channel], [0], None, [8], [0, 256]).ravel()
            hist = hist / max(1.0, float(hist.sum()))
            features.extend(hist.astype(float).tolist())

    features.extend(
        [
            float(gray.mean()),
            float(gray.std()),
            float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            float((edges > 0).mean()),
            float(np.percentile(gray, 5)),
            float(np.percentile(gray, 50)),
            float(np.percentile(gray, 95)),
        ]
    )
    return np.asarray(features, dtype=np.float32)
