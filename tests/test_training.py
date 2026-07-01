import numpy as np

from src.models import CNN_MODEL_SPECS, predict_probabilities, save_model, set_reproducible_seed, train_baseline


def test_baseline_training_and_probability_shape(work_dir):
    set_reproducible_seed(123)
    x = np.asarray([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.float32)
    y = np.asarray([0, 2, 1, 2], dtype=int)
    bundle, elapsed = train_baseline(x, y, n_estimators=8, seed=123)
    assert elapsed >= 0
    probs = predict_probabilities(bundle, [0.2, 0.1])
    assert probs.shape == (5,)
    np.testing.assert_allclose(probs.sum(), 1.0)
    assert save_model(bundle, work_dir / "baseline.pkl").exists()


def test_transfer_model_registry_contains_required_models():
    assert {"efficientnet_b0", "efficientnet_b3", "resnet50"}.issubset(CNN_MODEL_SPECS)