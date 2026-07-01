from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .constants import DR_CLASSES
from .gradcam import GradCamUnavailable, generate_keras_gradcam, save_unavailable_explanation
from .models import load_model, predict_image_probabilities, predict_probabilities
from .preprocessing import extract_handcrafted_features
from .quality_check import assess_quality, load_quality_thresholds
from .report_generator import generate_report
from .uncertainty import load_uncertainty_thresholds, route_case


def _recommendation(quality: dict, prediction: dict, uncertainty: dict) -> dict:
    if quality.get("status") != "accepted":
        return {
            "urgency": "repeat imaging",
            "text": "Image quality did not pass the safety gate. Repeat acquisition or route the case to manual clinical review before relying on model output.",
        }
    if uncertainty.get("manual_review"):
        return {
            "urgency": "manual review",
            "text": "Model confidence or uncertainty thresholds require qualified clinical review before any care decision.",
        }
    class_id = prediction.get("class_id")
    if class_id is None:
        return {
            "urgency": "manual review",
            "text": "No model prediction is available. Route this case for clinical review.",
        }
    if class_id >= 3:
        return {
            "urgency": "urgent ophthalmology review",
            "text": "The screening output suggests severe diabetic retinopathy or proliferative disease. Arrange prompt specialist review.",
        }
    if class_id == 2:
        return {
            "urgency": "ophthalmology review",
            "text": "The screening output suggests moderate diabetic retinopathy. Schedule clinician review and correlate with patient history.",
        }
    if class_id == 1:
        return {
            "urgency": "routine follow-up",
            "text": "The screening output suggests mild diabetic retinopathy. Continue routine clinical review and follow-up planning.",
        }
    return {
        "urgency": "routine screening follow-up",
        "text": "The screening output does not suggest diabetic retinopathy. Continue routine screening per clinical guidance.",
    }


def _active_model_path(model_path: Path, fallback_model_path: Path | None) -> tuple[Path | None, bool, str]:
    if model_path.exists():
        return model_path, False, "requested"
    if fallback_model_path and fallback_model_path.exists():
        return fallback_model_path, True, "fallback_baseline"
    return None, False, "model_missing"


def screen_retina_image(
    image_path: str | Path,
    model_path: str | Path = "models/efficientnet_b0.keras",
    thresholds_path: str | Path = "configs/thresholds.yaml",
    output_dir: str | Path = "reports/sample_reports",
    fallback_model_path: str | Path | None = "models/baseline_sklearn.pkl",
    patient_id: str | None = None,
) -> dict:
    image_path = Path(image_path)
    model_path = Path(model_path)
    fallback_path = Path(fallback_model_path) if fallback_model_path else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now().isoformat(timespec="seconds")
    quality = assess_quality(image_path, load_quality_thresholds(thresholds_path)).to_dict()
    probabilities = None
    unavailable_reason = "model_missing"
    latency_ms = None
    prediction = {
        "status": "not_run",
        "class_id": None,
        "class_name": None,
        "confidence": None,
        "probabilities": None,
    }
    model_info = {
        "requested_path": str(model_path),
        "active_path": None,
        "kind": None,
        "fallback_mode": False,
        "reason": None,
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
            if bundle.get("kind") == "keras_cnn":
                probabilities = predict_image_probabilities(bundle, image_path)
            else:
                features = extract_handcrafted_features(image_path)
                probabilities = predict_probabilities(bundle, features)
            latency_ms = round((perf_counter() - started) * 1000, 3)
            class_id = int(probabilities.argmax())
            prediction = {
                "status": "available",
                "class_id": class_id,
                "class_name": DR_CLASSES[class_id],
                "confidence": round(float(probabilities.max()), 6),
                "probabilities": [round(float(value), 6) for value in probabilities.tolist()],
            }
            if fallback_mode:
                prediction["status"] = "available_fallback_baseline"
        except Exception as exc:  # ponytail: corrupt/mismatched model artifacts route to review, not a crashed app.
            unavailable_reason = "model_error"
            model_info["reason"] = "model_error"
            prediction = {
                **prediction,
                "status": "model_error",
                "error": str(exc),
            }
    elif quality["status"] != "accepted":
        unavailable_reason = "quality_rejected"
        prediction["status"] = "skipped_quality_rejected"
        model_info["reason"] = "quality_rejected"
    else:
        prediction["status"] = "model_missing"
        model_info["reason"] = "model_missing"

    uncertainty = route_case(
        quality["status"],
        probabilities,
        load_uncertainty_thresholds(thresholds_path),
        unavailable_reason=unavailable_reason,
    )
    recommendation = _recommendation(quality, prediction, uncertainty)
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"
    gradcam_path = output_dir / f"{run_id}_gradcam.png"

    if prediction["status"] in {"available", "available_fallback_baseline"} and active_model is not None:
        try:
            bundle = load_model(active_model)
            if bundle.get("kind") != "keras_cnn":
                save_unavailable_explanation(image_path, gradcam_path, "Grad-CAM requires CNN checkpoint; baseline used")
            else:
                metadata = bundle.get("metadata", {})
                generate_keras_gradcam(
                    bundle["model"],
                    image_path,
                    metadata.get("last_conv_layer"),
                    gradcam_path,
                    class_index=prediction["class_id"],
                    size=int(metadata.get("input_size", 224)),
                    model_name=metadata.get("model_name", "efficientnet_b0"),
                )
        except (GradCamUnavailable, Exception) as exc:
            save_unavailable_explanation(image_path, gradcam_path, f"Grad-CAM unavailable: {exc}")
    else:
        save_unavailable_explanation(image_path, gradcam_path)

    result = {
        "metadata": {
            "run_id": run_id,
            "patient_id": patient_id,
            "generated_at": generated_at,
        },
        "preprocessing": {
            "crop_black_borders": True,
            "image_size": 224,
        },
        "quality": quality,
        "model": model_info,
        "prediction": prediction,
        "uncertainty": uncertainty,
        "recommendation": recommendation,
        "outputs": {
            "gradcam_path": str(gradcam_path),
            "explanation_path": str(gradcam_path),
            "report_path": None,
        },
        "performance": {
            "latency_ms": latency_ms,
        },
    }
    report_path = output_dir / f"{run_id}_ai_screening_report.pdf"
    result["outputs"]["report_path"] = str(generate_report(result, image_path, report_path, patient_id=patient_id))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RetinaAI screening inference on one image.")
    parser.add_argument("--image", required=True, help="Path to retinal image")
    parser.add_argument("--model", default="models/efficientnet_b0.keras")
    parser.add_argument("--fallback-model", default="models/baseline_sklearn.pkl")
    parser.add_argument("--thresholds", default="configs/thresholds.yaml")
    parser.add_argument("--output-dir", default="reports/sample_reports")
    parser.add_argument("--patient-id", default=None)
    args = parser.parse_args()

    result = screen_retina_image(
        args.image,
        model_path=args.model,
        fallback_model_path=args.fallback_model,
        thresholds_path=args.thresholds,
        output_dir=args.output_dir,
        patient_id=args.patient_id,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())