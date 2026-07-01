import numpy as np
import pytest

from src.preprocessing import crop_black_borders, extract_handcrafted_features, load_image, preprocess_image, resize_image


def test_preprocess_load_resize_and_features(synthetic_retina):
    image = load_image(synthetic_retina)
    assert image.shape == (360, 360, 3)
    resized = resize_image(image, 224)
    assert resized.shape == (224, 224, 3)
    processed = preprocess_image(synthetic_retina, size=128)
    assert processed.shape == (128, 128, 3)
    features = extract_handcrafted_features(synthetic_retina)
    assert features.ndim == 1
    assert features.size > 20
    assert np.isfinite(features).all()


def test_crop_black_borders_keeps_nonempty_image(synthetic_retina):
    cropped = crop_black_borders(synthetic_retina)
    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0
    assert cropped.shape[2] == 3


def test_load_image_rejects_bad_shape():
    with pytest.raises(ValueError):
        load_image(np.zeros((10, 10, 2), dtype=np.uint8))