from pathlib import Path

import numpy as np
import pytest

from src.gradcam import GradCamUnavailable, find_last_conv_layer, overlay_heatmap


def test_overlay_heatmap_saves_image(synthetic_retina, work_dir):
    path = overlay_heatmap(synthetic_retina, np.ones((16, 16), dtype=np.float32), work_dir / "overlay.png")
    assert Path(path).exists()


def test_overlay_heatmap_requires_2d_heatmap(synthetic_retina, work_dir):
    with pytest.raises(ValueError):
        overlay_heatmap(synthetic_retina, np.ones((4, 4, 3), dtype=np.float32), work_dir / "bad.png")


def test_find_last_conv_layer_reports_unavailable():
    class EmptyModel:
        layers = []

    with pytest.raises(GradCamUnavailable):
        find_last_conv_layer(EmptyModel())