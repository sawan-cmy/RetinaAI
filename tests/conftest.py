from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def work_dir() -> Path:
    path = Path("tests") / "_generated" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def synthetic_retina(work_dir: Path) -> Path:
    image = np.zeros((360, 360, 3), dtype=np.uint8)
    cv2.circle(image, (180, 180), 145, (92, 42, 25), -1)
    cv2.circle(image, (132, 168), 26, (210, 165, 120), -1)
    for offset in range(-90, 100, 30):
        cv2.line(image, (132, 168), (260, 180 + offset), (160, 55, 45), 3)
        cv2.line(image, (132, 168), (70, 170 + offset // 2), (145, 45, 38), 2)
    noise = np.random.default_rng(42).normal(0, 6, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    path = work_dir / "synthetic_retina.png"
    Image.fromarray(image).save(path)
    return path


@pytest.fixture
def dark_image(work_dir: Path) -> Path:
    path = work_dir / "dark.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(path)
    return path