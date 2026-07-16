from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from PIL import Image

from .constants import DR_CLASSES
from .gradcam import GradCamUnavailable, generate_keras_gradcam, generate_torch_gradcam, save_unavailable_explanation
from .models import load_model, predict_image_probabilities, predict_probabilities
from .preprocessing import extract_handcrafted_features, preprocess_image
from .quality_check import assess_quality, load_quality_thresholds
from .report_generator import REPORT_VERSION, build_report_data, generate_report
from .uncertainty import load_uncertainty_thresholds, normalize_probabilities, route_case


PATIENT_FIELDS = (
    "patient_id",
    "age",
    "sex",
    "diabetes_type",
    "known_duration_of_diabetes",
    "latest_hba1c",
    "blood_pressure",
    "previous_dr_history",
    "current_visual_symptoms",
)
ACQUISITION_FIELDS = ("eye_laterality", "capture_device", "screening_site", "operator_id")


def _sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_reason(existing: str | None, reason: str) -> str:
    reasons = [item for item in str(existing or "").split(",") if item]
    if reason not in reasons:
        reasons.append(reason)
    return ",".join(reasons)


def _metadata_subset(values: dict | None, fields: tuple[str, ...]) -> dict:
    values = values or {}
    return {field: values.get(field) for field in fields if values.get(field) is not None}


def _patient_payload(patient_id: str | None, patient_metadata: dict | None) -> dict:
    payload = _metadata_subset(patient_metadata, PATIENT_FIELDS)
    if patient_id and not payload.get("patient_id"):
        payload["patient_id"] = patient_id
    return payload


def _acquisition_payload(site_id: str | None, acquisition_metadata: dict | None) -> dict:
    payload = _metadata_subset(acquisition_metadata, ACQUISITION_FIELDS)
    if site_id and not payload.get("screening_site"):
        payload["screening_site"] = site_id
    return payload


def _recommendation(quality: dict, prediction: dict, uncertainty: dict, *, fallback_mode: bool = False, model_reason: str | None = None) -> dict:
    if quality.get("status") != "accepted":
        return {
            "urgency": "repeat imaging or manual assessment",
            "text": "Image quality did not pass the safety gate. Repeat acquisition or route the case to manual clinical review before relying on model output.",
        }
    if fallback_mode:
        return {
            "urgency": "manual review",
            "text": "Primary validated model unavailable. Fallback output must not be used for automated screening disposition. Manual review is required.",
        }
    if prediction.get("status") in {"model_missing", "model_error", "not_run"} or model_reason in {"model_missing", "model_error"}:
        return {"urgency": "manual review", "text": "No validated primary model prediction is available. Route this case for clinical review."}
    if uncertainty.get("manual_review"):
        return {"urgency": "manual review", "text": "Model confidence or uncertainty thresholds require qualified clinical review before any care decision."}
    class_id = prediction.get("class_id")
    if class_id is None:
        return {"urgency": "manual review", "text": "No model prediction is available. Route this case for clinical review."}
    if class_id >= 3:
        return {"urgency": "priority ophthalmology assessment", "text": "Severe or proliferative DR is suspected by this screening model. Priority ophthalmology assessment is recommended."}
    if class_id == 2:
        return {"urgency": "ophthalmology review", "text": "Referable DR is suspected by this screening model. Ophthalmology review is recommended."}
    if class_id == 1:
        return {"urgency": "clinician review", "text": "Mild DR screening features are suspected. Clinician review and follow-up planning are required."}
    return {"urgency": "clinician protocol", "text": "No apparent DR was detected by this screening model. Continue screening according to the treating clinician's protocol; absence of disease is not guaranteed."}


def _active_model_path(model_path: Path, fallback_model_path: Path | None) -> tuple[Path | None, bool, str]:
    if model_path.exists():
        return model_path, False, "requested"
    if fallback_model_path and fallback_model_path.exists():
        return fallback_model_path, True, "fallback_baseline"
    return None, False, "model_missing"


def screen_retina_image(
    image_path: str | Path,
    model_path: str | Path = "models/efficientnet_b0_torch_transfer_acc.pt",
    thresholds_path: str | Path = "configs/thresholds.yaml",
    output_dir: str | Path = "reports/sample_reports",
    fallback_model_path: str | Path | None = "models/baseline_sklearn.pkl",
    patient_id: str | None = None,
    site_id: str | None = None,
    patient_metadata: dict | None = None,
    acquisition_metadata: dict | None = None,
) -> dict:
    image_path = Path(image_path)
    model_path = Path(model_path)
    thresholds_path = Path(thresholds_path)
    fallback_path = Path(fallback_model_path) if fallback_model_path else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().isoformat(timespec="seconds")
    quality_thresholds = load_quality_thresholds(thresholds_path, site_id=site_id)
    uncertainty_thresholds = load_uncertainty_thresholds(thresholds_path, site_id=site_id)
    quality = assess_quality(image_path, quality_thresholds).to_dict()
    probabilities = None
    unavailable_reason = "model_missing"
    latency_ms = None
    prediction = {"status": "not_run", "class_id": None, "class_name": None, "confidence": None, "probabilities": None}
    model_info = {
        "requested_path": str(model_path),
        "active_path": None,
        "kind": None,
        "fallback_mode": False,
        "reason": None,
        "metadata": {},
        "thresholds_path": str(thresholds_path),
    }

    active_model, fallback_mode, model_reason = _active_model_path(model_path, fallback_path)
    model_info["fallback_mode"] = fallback_mode
    model_info["reason"] = model_reason
    if active_model is not None:
        model_info["active_path"] = str(active_model)

    if quality["status"] == "accepted" and active_model is not None:
        try:
            started = perf_counter()
            bundle = load_model(active_model)
            model_info["kind"] = bundle.get("kind", "unknown")
            model_info["metadata"] = dict(bundle.get("metadata") or {})
            if bundle.get("kind") in {"keras_cnn", "torch_cnn"}:
                probabilities = predict_image_probabilities(bundle, image_path)
            else:
                probabilities = predict_probabilities(bundle, extract_handcrafted_features(image_path))
            probabilities = normalize_probabilities(probabilities)
            if probabilities.size != len(DR_CLASSES):
                raise ValueError(f"expected {len(DR_CLASSES)} class probabilities, got {probabilities.size}")
            latency_ms = round((perf_counter() - started) * 1000, 3)
            class_id = int(np.argmax(probabilities))
            prediction = {
                "status": "available_fallback_baseline" if fallback_mode else "available",
                "class_id": class_id,
                "class_name": DR_CLASSES[class_id],
                "confidence": round(float(probabilities.max()), 6),
                "probabilities": [round(float(value), 6) for value in probabilities.tolist()],
            }
        except Exception as exc:  # ponytail: corrupt/mismatched model artifacts route to review, not a crashed app.
            probabilities = None
            unavailable_reason = "model_error"
            model_info["reason"] = "model_error"
            prediction = {**prediction, "status": "model_error", "error": str(exc)}
    elif quality["status"] != "accepted":
        unavailable_reason = "quality_rejected"
        prediction["status"] = "skipped_quality_rejected"
        model_info["reason"] = "quality_rejected"
    else:
        prediction["status"] = "model_missing"
        model_info["reason"] = "model_missing"

    try:
        uncertainty = route_case(quality["status"], probabilities, uncertainty_thresholds, unavailable_reason=unavailable_reason)
    except ValueError:
        probabilities = None
        prediction = {**prediction, "status": "model_error", "error": "invalid probability vector"}
        model_info["reason"] = "model_error"
        uncertainty = route_case(quality["status"], None, uncertainty_thresholds, unavailable_reason="model_error")
    if fallback_mode:
        uncertainty["manual_review"] = True
        uncertainty["reason"] = _append_reason(uncertainty.get("reason"), "fallback_model_used")

    recommendation = _recommendation(quality, prediction, uncertainty, fallback_mode=fallback_mode, model_reason=model_info.get("reason"))
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    gradcam_path = output_dir / f"{run_id}_gradcam.png"
    processed_path = output_dir / f"{run_id}_model_input.png"
    report_path = output_dir / f"{run_id}_ai_screening_report.pdf"
    json_path = report_path.with_suffix(".json")

    processed_image_path: str | None = None
    try:
        Image.fromarray(preprocess_image(image_path, size=224)).save(processed_path)
        processed_image_path = str(processed_path)
    except Exception:
        processed_image_path = None

    gradcam_status = "unavailable"
    gradcam_message = "Grad-CAM was not generated for this run."
    if prediction["status"] in {"available", "available_fallback_baseline"} and active_model is not None:
        try:
            bundle = load_model(active_model)
            metadata = bundle.get("metadata", {})
            if bundle.get("kind") == "keras_cnn":
                generate_keras_gradcam(bundle["model"], image_path, metadata.get("last_conv_layer"), gradcam_path, class_index=prediction["class_id"], size=int(metadata.get("input_size", 224)), model_name=metadata.get("model_name", "efficientnet_b0"))
                gradcam_status = "available"
                gradcam_message = "Grad-CAM overlay generated for CNN model output."
            elif bundle.get("kind") == "torch_cnn":
                generate_torch_gradcam(bundle["model"], image_path, metadata.get("last_conv_layer"), gradcam_path, class_index=prediction["class_id"], size=int(metadata.get("input_size", 224)))
                gradcam_status = "available"
                gradcam_message = "Grad-CAM overlay generated for CNN model output."
            else:
                save_unavailable_explanation(image_path, gradcam_path, "Grad-CAM requires CNN checkpoint; baseline used")
                gradcam_message = "Grad-CAM requires a CNN checkpoint; fallback baseline was used."
        except (GradCamUnavailable, Exception):
            save_unavailable_explanation(image_path, gradcam_path, "Grad-CAM unavailable for this run")
            gradcam_message = "Grad-CAM could not be generated for this run."
    else:
        save_unavailable_explanation(image_path, gradcam_path)

    result = {
        "report_version": REPORT_VERSION,
        "metadata": {
            "run_id": run_id,
            "report_id": run_id,
            "report_version": REPORT_VERSION,
            "patient_id": patient_id,
            "site_id": site_id,
            "generated_at": generated_at,
        },
        "patient": _patient_payload(patient_id, patient_metadata),
        "acquisition": _acquisition_payload(site_id, acquisition_metadata),
        "preprocessing": {"crop_black_borders": True, "image_size": 224, "processed_image_path": processed_image_path},
        "quality": quality,
        "quality_thresholds": quality_thresholds,
        "model": model_info,
        "prediction": prediction,
        "uncertainty": uncertainty,
        "uncertainty_thresholds": uncertainty_thresholds,
        "thresholds": {"path": str(thresholds_path), "quality": quality_thresholds, "uncertainty": uncertainty_thresholds},
        "recommendation": recommendation,
        "model_provenance": {
            "checkpoint_sha256": _sha256(active_model),
            "threshold_sha256": _sha256(thresholds_path),
            "fallback_used": fallback_mode,
        },
        "outputs": {
            "gradcam_path": str(gradcam_path),
            "explanation_path": str(gradcam_path),
            "processed_image_path": processed_image_path,
            "report_path": str(report_path),
            "json_path": str(json_path),
            "gradcam_status": gradcam_status,
            "gradcam_message": gradcam_message,
        },
        "performance": {"latency_ms": latency_ms},
    }
    result["outputs"]["report_path"] = str(generate_report(result, image_path, report_path, patient_id=patient_id))
    structured = build_report_data(result, image_path)
    result["report_version"] = structured["report_version"]
    result["metadata"].update({"report_version": structured["metadata"]["report_version"], "software_report_version": structured["metadata"]["software_report_version"]})
    for key in ("patient", "acquisition", "image_quality", "dr_grading", "clinical_endpoints", "lesion_analysis", "referral", "clinician_review"):
        result[key] = structured[key]
    result["uncertainty"].update(structured["uncertainty"])
    result["model_provenance"].update(structured["model_provenance"])
    result["performance"].update(structured["performance"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RetinaAI screening inference on one image.")
    parser.add_argument("--image", required=True, help="Path to retinal image")
    parser.add_argument("--model", default="models/efficientnet_b0_torch_transfer_acc.pt")
    parser.add_argument("--fallback-model", default="models/baseline_sklearn.pkl")
    parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    parser.add_argument("--output-dir", default="reports/sample_reports")
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--site-id", default=None)
    parser.add_argument("--age", default=None)
    parser.add_argument("--sex", default=None)
    parser.add_argument("--eye-laterality", choices=["left", "right", "unknown"], default=None)
    parser.add_argument("--screening-site", default=None)
    parser.add_argument("--operator-id", default=None)
    args = parser.parse_args()

    result = screen_retina_image(
        args.image,
        model_path=args.model,
        fallback_model_path=args.fallback_model,
        thresholds_path=args.thresholds,
        output_dir=args.output_dir,
        patient_id=args.patient_id,
        site_id=args.site_id,
        patient_metadata={"age": args.age, "sex": args.sex},
        acquisition_metadata={"eye_laterality": args.eye_laterality, "screening_site": args.screening_site, "operator_id": args.operator_id},
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
