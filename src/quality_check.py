from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .preprocessing import load_image

DEFAULT_THRESHOLDS = {
    "min_blur": 20.0,
    "min_brightness": 25.0,
    "max_brightness": 235.0,
    "min_contrast": 10.0,
    "min_retina_visibility": 0.20,
}


@dataclass(frozen=True)
class QualityResult:
    status: str
    blur_score: float
    brightness_score: float
    contrast_score: float
    retina_visibility_score: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _site_section(data: dict, site_id: str | None, section: str) -> dict:
    if not site_id:
        return {}
    sites = data.get("sites", {}) or {}
    site = sites.get(site_id) or sites.get(site_id.lower()) or {}
    return site.get(section, site) or {}


def load_quality_thresholds(path: str | Path | None = None, site_id: str | None = None) -> dict:
    if not path or not Path(path).exists():
        return DEFAULT_THRESHOLDS.copy()
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    base = data.get("quality", data) or {}
    return {**DEFAULT_THRESHOLDS, **base, **_site_section(data, site_id, "quality")}


def retina_visibility_score(image_or_path: str | Path | np.ndarray) -> float:
    image = load_image(image_or_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    radius = min(height, width) * 0.45
    circle = (yy - height / 2) ** 2 + (xx - width / 2) ** 2 <= radius**2
    foreground = (gray > 15) & (hsv[:, :, 1] > 5)
    return float(foreground[circle].mean()) if circle.any() else 0.0


def assess_quality(image_or_path: str | Path | np.ndarray, thresholds: dict | None = None) -> QualityResult:
    image = load_image(image_or_path)
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    visibility = retina_visibility_score(image)

    reasons: list[str] = []
    if blur < float(thresholds["min_blur"]):
        reasons.append("blur_too_high")
    if brightness < float(thresholds["min_brightness"]):
        reasons.append("image_too_dark")
    if brightness > float(thresholds["max_brightness"]):
        reasons.append("image_too_bright")
    if contrast < float(thresholds["min_contrast"]):
        reasons.append("contrast_too_low")
    if visibility < float(thresholds["min_retina_visibility"]):
        reasons.append("retina_visibility_too_low")

    return QualityResult(
        status="rejected" if reasons else "accepted",
        blur_score=round(blur, 3),
        brightness_score=round(brightness, 3),
        contrast_score=round(contrast, 3),
        retina_visibility_score=round(visibility, 3),
        reasons=reasons,
    )