import numpy as np
import pytest

from src.models import _build_torch_model, load_model, predict_image_probabilities


def test_torch_checkpoint_loads_and_predicts(synthetic_retina, work_dir):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")

    model, _ = _build_torch_model("efficientnet_b0")
    checkpoint_path = work_dir / "efficientnet.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "arch": "efficientnet_b0",
            "model_name": "efficientnet_b0_torch_test",
        },
        checkpoint_path,
    )

    bundle = load_model(checkpoint_path)
    assert bundle["kind"] == "torch_cnn"
    probabilities = predict_image_probabilities(bundle, synthetic_retina)
    assert probabilities.shape == (5,)
    np.testing.assert_allclose(probabilities.sum(), 1.0, rtol=1e-5)