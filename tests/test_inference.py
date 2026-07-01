from pathlib import Path

import numpy as np

from src.inference import screen_retina_image
from src.models import save_model, train_baseline
from src.preprocessing import extract_handcrafted_features


def _tiny_baseline_model(image_path: Path, model_path: Path) -> Path:
    features = extract_handcrafted_features(image_path)
    x = np.vstack([features, features + 0.01, features + 0.02])
    y = np.asarray([0, 1, 2], dtype=int)
    bundle, _ = train_baseline(x, y, n_estimators=8, seed=7)
    return save_model(bundle, model_path)


def test_inference_missing_model_does_not_crash(synthetic_retina, work_dir):
    result = screen_retina_image(synthetic_retina, model_path=work_dir / "missing.keras", fallback_model_path=None, output_dir=work_dir / "reports")
    assert result["prediction"]["status"] == "model_missing"
    assert result["uncertainty"]["manual_review"] is True
    assert Path(result["outputs"]["report_path"]).exists()
    assert Path(result["outputs"]["gradcam_path"]).exists()


def test_inference_uses_baseline_fallback(synthetic_retina, work_dir):
    fallback = _tiny_baseline_model(synthetic_retina, work_dir / "baseline.pkl")
    result = screen_retina_image(synthetic_retina, model_path=work_dir / "missing.keras", fallback_model_path=fallback, output_dir=work_dir / "reports")
    assert result["model"]["fallback_mode"] is True
    assert result["prediction"]["status"] == "available_fallback_baseline"
    assert len(result["prediction"]["probabilities"]) == 5