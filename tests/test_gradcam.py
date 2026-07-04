from pathlib import Path

import numpy as np
import pytest

from src.gradcam import GradCamUnavailable, find_last_conv_layer, find_last_torch_conv_layer, generate_torch_gradcam, overlay_heatmap


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


def test_generate_torch_gradcam_saves_image(synthetic_retina, work_dir):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    class TinyTorchCnn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
            self.head = torch.nn.Linear(4, 5)

        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = self.pool(x).flatten(1)
            return self.head(x)

    model = TinyTorchCnn()
    assert find_last_torch_conv_layer(model) == "conv"
    path = generate_torch_gradcam(model, synthetic_retina, None, work_dir / "torch_gradcam.png", class_index=0, size=64)
    assert Path(path).exists()
