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
    assert result["uncertainty"]["manual_review"] is True
    assert result["referral"]["result"] == "Not validated for automated disposition"


def test_inference_uses_torch_gradcam_for_pt_models(synthetic_retina, work_dir, monkeypatch):
    model_path = work_dir / "model.pt"
    model_path.write_bytes(b"placeholder")
    calls = []
    fake_model = object()

    def fake_load_model(_path):
        return {
            "kind": "torch_cnn",
            "model": fake_model,
            "metadata": {"input_size": 96, "last_conv_layer": "features.0"},
        }

    def fake_generate_torch_gradcam(model, image_path, last_conv_layer_name, output_path, class_index=None, size=224):
        calls.append((model, image_path, last_conv_layer_name, class_index, size))
        Path(output_path).write_bytes(b"torch gradcam")
        return Path(output_path)

    monkeypatch.setattr("src.inference.load_model", fake_load_model)
    monkeypatch.setattr("src.inference.predict_image_probabilities", lambda _bundle, _image: np.asarray([0.91, 0.03, 0.02, 0.02, 0.02]))
    monkeypatch.setattr("src.inference.generate_torch_gradcam", fake_generate_torch_gradcam)

    result = screen_retina_image(synthetic_retina, model_path=model_path, fallback_model_path=None, output_dir=work_dir / "reports")

    assert result["model"]["kind"] == "torch_cnn"
    assert result["prediction"]["status"] == "available"
    assert calls == [(fake_model, synthetic_retina, "features.0", 0, 96)]


def test_inference_creates_pdf_json_sidecar_and_structured_schema(synthetic_retina, work_dir):
    result = screen_retina_image(
        synthetic_retina,
        model_path=work_dir / "missing.keras",
        fallback_model_path=None,
        output_dir=work_dir / "reports",
        patient_id="patient-structured",
    )
    assert result["report_version"] == "2.0"
    assert result["patient"]["patient_id"] == "patient-structured"
    assert result["clinical_endpoints"]["referable_dr"]["status"] == "indeterminate"
    assert Path(result["outputs"]["report_path"]).exists()
    assert Path(result["outputs"]["json_path"]).exists()
    assert Path(result["outputs"]["processed_image_path"]).exists()


def test_inference_rejects_poor_quality_without_definitive_classification(dark_image, work_dir):
    result = screen_retina_image(dark_image, model_path=work_dir / "missing.keras", fallback_model_path=None, output_dir=work_dir / "reports")
    assert result["quality"]["status"] == "rejected"
    assert result["prediction"]["status"] == "skipped_quality_rejected"
    assert result["referral"]["manual_review_required"] is True
    assert result["clinical_endpoints"]["referable_dr"]["status"] == "indeterminate"


def test_inference_model_exception_routes_to_manual_review(synthetic_retina, work_dir):
    corrupt = work_dir / "corrupt.pkl"
    corrupt.write_bytes(b"not a pickle")
    result = screen_retina_image(synthetic_retina, model_path=corrupt, fallback_model_path=None, output_dir=work_dir / "reports")
    assert result["prediction"]["status"] == "model_error"
    assert result["uncertainty"]["manual_review"] is True
    assert result["referral"]["manual_review_required"] is True


def test_inference_high_uncertainty_prediction_routes_to_review(synthetic_retina, work_dir, monkeypatch):
    model_path = work_dir / "model.pt"
    model_path.write_bytes(b"placeholder")

    monkeypatch.setattr("src.inference.load_model", lambda _path: {"kind": "torch_cnn", "model": object(), "metadata": {"input_size": 96}})
    monkeypatch.setattr("src.inference.predict_image_probabilities", lambda _bundle, _image: np.asarray([0.3, 0.25, 0.2, 0.15, 0.1]))
    monkeypatch.setattr("src.inference.generate_torch_gradcam", lambda *_args, **_kwargs: Path(_args[3]).write_bytes(b"gradcam"))

    result = screen_retina_image(synthetic_retina, model_path=model_path, fallback_model_path=None, output_dir=work_dir / "reports")
    assert result["prediction"]["status"] == "available"
    assert result["uncertainty"]["manual_review"] is True
    assert "low_confidence" in result["uncertainty"]["reason"]
    assert result["clinical_endpoints"]["referable_dr"]["status"] == "indeterminate"


def test_inference_malformed_probabilities_route_to_model_error(synthetic_retina, work_dir, monkeypatch):
    model_path = work_dir / "model.pt"
    model_path.write_bytes(b"placeholder")

    monkeypatch.setattr("src.inference.load_model", lambda _path: {"kind": "torch_cnn", "model": object(), "metadata": {"input_size": 96}})
    monkeypatch.setattr("src.inference.predict_image_probabilities", lambda _bundle, _image: np.asarray([float("nan"), 0.2, 0.3, 0.2, 0.3]))

    result = screen_retina_image(synthetic_retina, model_path=model_path, fallback_model_path=None, output_dir=work_dir / "reports")
    assert result["prediction"]["status"] == "model_error"
    assert result["uncertainty"]["manual_review"] is True
