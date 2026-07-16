from __future__ import annotations

import json
from pathlib import Path

import pytest
from pypdf import PdfReader

from src.report_generator import build_report_data, generate_report


def _text(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _base_result(class_id: int = 0, probabilities: list[float] | None = None) -> dict:
    probabilities = [0.02, 0.02, 0.02, 0.02, 0.92] if probabilities is None else probabilities
    if class_id != 4 and probabilities:
        probabilities = [0.02] * 5
        probabilities[class_id] = 0.92
    confidence = max(probabilities) if probabilities else None
    return {
        "metadata": {"run_id": "run-test", "generated_at": "2026-07-14T10:00:00", "patient_id": "patient-1", "site_id": "clinic-a"},
        "patient": {"patient_id": "patient-1"},
        "acquisition": {"eye_laterality": "left", "screening_site": "clinic-a", "operator_id": "operator-1"},
        "preprocessing": {"crop_black_borders": True, "image_size": 224},
        "quality": {"status": "accepted", "blur_score": 100.0, "brightness_score": 80.0, "contrast_score": 25.0, "retina_visibility_score": 0.8, "reasons": []},
        "quality_thresholds": {"min_blur": 20.0, "min_brightness": 25.0, "max_brightness": 235.0, "min_contrast": 10.0, "min_retina_visibility": 0.2},
        "prediction": {"status": "available", "class_id": class_id, "class_name": f"class {class_id}", "confidence": confidence, "probabilities": probabilities},
        "uncertainty": {"manual_review": False, "reason": "confidence_above_threshold", "confidence": confidence, "entropy": 0.35, "margin": 0.9},
        "uncertainty_thresholds": {"min_confidence": 0.7, "max_entropy": 1.2, "min_top2_margin": 0.15},
        "model": {"kind": "torch_cnn", "fallback_mode": False, "reason": "requested", "active_path": "C:\\internal\\models\\primary.pt", "metadata": {"model_name": "efficientnet_b0", "arch": "efficientnet_b0"}},
        "outputs": {"gradcam_status": "unavailable", "gradcam_message": "Grad-CAM was not generated for this run."},
        "performance": {"latency_ms": 12.3},
    }


@pytest.mark.parametrize(
    ("class_id", "referable", "sight", "category"),
    [
        (0, "negative", "negative", "Continue screening"),
        (1, "negative", "negative", "Clinician review"),
        (2, "positive", "negative", "Ophthalmology review"),
        (3, "positive", "positive", "Priority ophthalmology"),
        (4, "positive", "positive", "Priority ophthalmology"),
    ],
)
def test_report_data_referral_mapping_each_grade(synthetic_retina, class_id, referable, sight, category):
    data = build_report_data(_base_result(class_id), synthetic_retina)
    assert data["clinical_endpoints"]["referable_dr"]["status"] == referable
    assert data["clinical_endpoints"]["sight_threatening_dr"]["status"] == sight
    assert category in data["referral"]["category"]


def test_report_handles_invalid_probabilities_and_backward_compatible_dict(synthetic_retina):
    result = _base_result(0, probabilities=[])
    result.pop("patient")
    result.pop("acquisition")
    data = build_report_data(result, synthetic_retina)
    assert data["patient"]["age"] == "Not provided"
    assert data["dr_grading"]["probabilities"] == []
    assert data["clinical_endpoints"]["referable_dr"]["status"] == "indeterminate"


def test_pdf_and_json_sidecar_include_safety_sections_without_path_leaks(synthetic_retina, work_dir):
    result = _base_result(2)
    result["patient"].update({"previous_dr_history": "x" * 700})
    result["acquisition"]["operator_id"] = "operator-" + "y" * 300
    pdf_path = work_dir / "retinaai_report.pdf"

    generated = generate_report(result, synthetic_retina, pdf_path)
    sidecar = generated.with_suffix(".json")
    assert generated.exists()
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["report_version"] == "2.0"
    assert "raw_image" not in json.dumps(data).lower()

    text = _text(generated)
    assert "Screening result only - not a confirmed diagnosis" in text
    assert "qualified medical professional" in text
    assert "Highlighted regions influenced the model output" in text
    assert "Microaneurysms" in text
    assert "Not evaluated by the current model" in text
    assert "Clinician Review Section" in text
    assert "Technical Audit Trail" in text
    assert "Traceback" not in text
    assert "C:\\" not in text


def test_pdf_handles_missing_gradcam_and_processed_image(synthetic_retina, work_dir):
    result = _base_result(0)
    result["outputs"] = {"gradcam_status": "unavailable", "gradcam_message": "Grad-CAM was not generated for this run."}
    pdf_path = generate_report(result, synthetic_retina, work_dir / "missing_images.pdf")
    text = _text(pdf_path)
    assert "Processed/cropped model-input image" in text
    assert "Grad-CAM availability" in text
    assert "Not available" in text


def test_uncertain_or_fallback_results_are_indeterminate(synthetic_retina):
    uncertain = _base_result(0, probabilities=[0.3, 0.25, 0.2, 0.15, 0.1])
    uncertain["uncertainty"] = {"manual_review": True, "reason": "low_confidence,high_entropy", "confidence": 0.3, "entropy": 1.55, "margin": 0.05}
    uncertain_data = build_report_data(uncertain, synthetic_retina)
    assert uncertain_data["clinical_endpoints"]["referable_dr"]["status"] == "indeterminate"
    assert uncertain_data["referral"]["manual_review_required"] is True

    fallback = _base_result(0)
    fallback["model"]["fallback_mode"] = True
    fallback["prediction"]["status"] = "available_fallback_baseline"
    fallback_data = build_report_data(fallback, synthetic_retina)
    assert fallback_data["referral"]["result"] == "Not validated for automated disposition"
    assert fallback_data["clinical_endpoints"]["any_dr"]["status"] == "indeterminate"
