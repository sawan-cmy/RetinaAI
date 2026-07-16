from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from reportlab.graphics.shapes import Drawing, Rect
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .constants import DISCLAIMER, PROJECT_ROOT

REPORT_VERSION = "2.0"
SOFTWARE_REPORT_VERSION = "RetinaAI report schema 2.0"
DR_SCREENING_LABELS = {
    0: "No apparent DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}
LESION_ROWS = ["Microaneurysms", "Retinal haemorrhages", "Hard exudates", "Soft exudates", "Neovascularization", "Macular involvement"]
ADDITIONAL_IMAGE_CHECKS = ["Optic-disc visibility", "Macula visibility", "Vessel visibility", "Reflection artifacts", "Eyelash artifacts", "Field-of-view completeness"]

NAVY = colors.HexColor("#0F2742")
BLUE = colors.HexColor("#1D4ED8")
TEAL = colors.HexColor("#0F766E")
GREEN = colors.HexColor("#15803D")
AMBER = colors.HexColor("#B45309")
RED = colors.HexColor("#B91C1C")
GREY = colors.HexColor("#64748B")
LIGHT_GREY = colors.HexColor("#F1F5F9")
BORDER = colors.HexColor("#CBD5E1")
TEXT = colors.HexColor("#111827")


def _clean_text(value: Any, default: str = "Not provided", limit: int = 500) -> str:
    if value is None or str(value).strip() == "":
        return default
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", str(value))
    text = re.sub(r"\s+", " ", text).strip()

    def split(match: re.Match[str]) -> str:
        return " ".join(textwrap.wrap(match.group(0), 38, break_long_words=True))

    text = re.sub(r"\S{39,}", split, text)
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text or default


def _p(value: Any, style: ParagraphStyle, default: str = "Not provided", limit: int = 500) -> Paragraph:
    return Paragraph(escape(_clean_text(value, default, limit)), style)


def _fmt(value: Any, default: str = "Not evaluated") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.3f}" if math.isfinite(value) else default
    return _clean_text(value, default)


def _pct(value: float | None) -> str:
    return "Not evaluated" if value is None else f"{value * 100:.1f}%"


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


def _safe_artifact_id(path: str | Path | None, default: str = "Not available") -> str:
    if not path:
        return default
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except Exception:
        return candidate.name or default


def _coerce_probabilities(probabilities: Any) -> list[float] | None:
    try:
        values = [float(value) for value in probabilities]
    except (TypeError, ValueError):
        return None
    if len(values) != len(DR_SCREENING_LABELS) or not all(math.isfinite(value) for value in values):
        return None
    total = sum(values)
    return None if total <= 0 else [value / total for value in values]


def _metric_row(label: str, value: Any, threshold: str, passed: bool | None, interpretation: str) -> dict[str, Any]:
    return {
        "metric": label,
        "value": _fmt(value),
        "threshold": threshold,
        "status": "Not evaluated" if passed is None else "Pass" if passed else "Fail",
        "interpretation": interpretation,
    }


def _quality_metrics(quality: dict[str, Any], thresholds: dict[str, Any]) -> list[dict[str, Any]]:
    def number(name: str) -> float | None:
        try:
            value = float(quality.get(name))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    blur = number("blur_score")
    brightness = number("brightness_score")
    contrast = number("contrast_score")
    visibility = number("retina_visibility_score")
    min_blur = float(thresholds.get("min_blur", 20.0))
    min_brightness = float(thresholds.get("min_brightness", 25.0))
    max_brightness = float(thresholds.get("max_brightness", 235.0))
    min_contrast = float(thresholds.get("min_contrast", 10.0))
    min_visibility = float(thresholds.get("min_retina_visibility", 0.20))
    return [
        _metric_row("Blur/sharpness score", quality.get("blur_score"), f">= {min_blur:g}", None if blur is None else blur >= min_blur, "Higher score means a sharper image for screening workflow use."),
        _metric_row("Brightness", quality.get("brightness_score"), f"{min_brightness:g} to {max_brightness:g}", None if brightness is None else min_brightness <= brightness <= max_brightness, "Mean image brightness should be inside the configured range."),
        _metric_row("Contrast", quality.get("contrast_score"), f">= {min_contrast:g}", None if contrast is None else contrast >= min_contrast, "Higher contrast helps the classifier receive usable retinal structure."),
        _metric_row("Retina visibility", quality.get("retina_visibility_score"), f">= {min_visibility:g}", None if visibility is None else visibility >= min_visibility, "Estimated visible retinal foreground within the central field."),
    ]


def _probability_rows(probabilities: list[float] | None) -> list[dict[str, Any]]:
    if probabilities is None:
        return []
    ranked = sorted(range(len(probabilities)), key=lambda index: probabilities[index], reverse=True)
    ranks = {class_id: rank + 1 for rank, class_id in enumerate(ranked)}
    return [
        {
            "class_id": class_id,
            "class_name": DR_SCREENING_LABELS[class_id],
            "probability": probabilities[class_id],
            "rank": ranks[class_id],
            "is_top": ranks[class_id] == 1,
            "is_second": ranks[class_id] == 2,
        }
        for class_id in range(len(DR_SCREENING_LABELS))
    ]


def _confidence_category(confidence: float | None, thresholds: dict[str, Any]) -> str:
    if confidence is None:
        return "Not evaluated"
    if confidence >= float(thresholds.get("min_confidence", 0.70)):
        return "High"
    return "Moderate" if confidence >= 0.50 else "Low"


def _append_reason(existing: str, reason: str) -> str:
    reasons = [item for item in existing.split(",") if item]
    if reason not in reasons:
        reasons.append(reason)
    return ",".join(reasons)


def _is_disposition_valid(quality_status: str, prediction: dict[str, Any], probabilities: list[float] | None, uncertainty: dict[str, Any], fallback_used: bool) -> bool:
    return (
        quality_status == "accepted"
        and prediction.get("status") == "available"
        and prediction.get("class_id") in DR_SCREENING_LABELS
        and probabilities is not None
        and not fallback_used
        and not bool(uncertainty.get("manual_review"))
    )


def _clinical_endpoints(valid: bool, class_id: int | None) -> dict[str, dict[str, Any]]:
    mapping = "Five-class DR screening grade only. This does not evaluate macular involvement or lesion-level findings."
    if not valid or class_id is None:
        reason = "Indeterminate - manual review required."
        return {
            "any_dr": {"status": "indeterminate", "value": None, "reason": reason, "mapping": mapping},
            "referable_dr": {"status": "indeterminate", "value": None, "reason": reason, "mapping": mapping},
            "sight_threatening_dr": {"status": "indeterminate", "value": None, "reason": reason, "mapping": mapping},
        }
    return {
        "any_dr": {"status": "positive" if class_id > 0 else "negative", "value": class_id > 0, "reason": "Derived from valid class > 0.", "mapping": mapping},
        "referable_dr": {"status": "positive" if class_id >= 2 else "negative", "value": class_id >= 2, "reason": "Derived from valid class >= 2.", "mapping": mapping},
        "sight_threatening_dr": {"status": "positive" if class_id >= 3 else "negative", "value": class_id >= 3, "reason": "Conservative mapping: severe NPDR or proliferative DR classes only.", "mapping": mapping},
    }


def _derive_referral(quality_status: str, prediction: dict[str, Any], uncertainty: dict[str, Any], fallback_used: bool, model_reason: str | None) -> dict[str, Any]:
    class_id = prediction.get("class_id")
    if quality_status != "accepted":
        return {"category": "Repeat image or manual clinical assessment", "result": "Indeterminate", "action": "Repeat image acquisition or obtain manual clinical assessment.", "manual_review_required": True, "color": "red", "rationale": "The image quality gate rejected the image."}
    if fallback_used:
        return {"category": "Manual review required - fallback model used", "result": "Not validated for automated disposition", "action": "Manual clinical review is required before any care decision.", "manual_review_required": True, "color": "amber", "rationale": "Primary validated model unavailable. Fallback output must not be used for automated screening disposition."}
    if prediction.get("status") in {"model_missing", "model_error", "not_run"} or model_reason in {"model_missing", "model_error"}:
        return {"category": "Manual review required - model unavailable", "result": "Not validated for automated disposition", "action": "Manual clinical review is required because model output is unavailable.", "manual_review_required": True, "color": "amber", "rationale": "No validated primary model prediction is available."}
    if uncertainty.get("manual_review"):
        return {"category": "Indeterminate - manual review required", "result": "Indeterminate", "action": "Manual clinical review is required before screening disposition.", "manual_review_required": True, "color": "amber", "rationale": f"Uncertainty routing triggered: {_clean_text(uncertainty.get('reason'), default='threshold failed')}."}
    if class_id == 0:
        return {"category": "Continue screening according to clinician protocol", "result": "No apparent DR detected by this screening model", "action": "Continue screening according to the treating clinician's protocol. This does not guarantee absence of disease.", "manual_review_required": False, "color": "green", "rationale": "Valid class 0 screening output passed quality and uncertainty gates."}
    if class_id == 1:
        return {"category": "Clinician review/follow-up planning required", "result": "Mild DR screening features suspected", "action": "Clinician review and follow-up planning are required.", "manual_review_required": True, "color": "amber", "rationale": "Valid class 1 screening output."}
    if class_id == 2:
        return {"category": "Ophthalmology review recommended", "result": "Referable DR suspected", "action": "Ophthalmology review is recommended.", "manual_review_required": True, "color": "amber", "rationale": "Valid class 2 screening output maps to referable DR suspicion."}
    if class_id in {3, 4}:
        return {"category": "Priority ophthalmology assessment recommended", "result": "Severe or proliferative DR suspected", "action": "Priority ophthalmology assessment is recommended.", "manual_review_required": True, "color": "red", "rationale": "Valid class 3 or 4 screening output."}
    return {"category": "Manual review required", "result": "Indeterminate", "action": "Manual clinical review is required before screening disposition.", "manual_review_required": True, "color": "amber", "rationale": "Prediction class was not available."}


def build_report_data(result: dict[str, Any], image_path: str | Path | None = None) -> dict[str, Any]:
    metadata = dict(result.get("metadata") or {})
    patient_in = dict(result.get("patient") or {})
    acquisition_in = dict(result.get("acquisition") or {})
    quality = dict(result.get("quality") or result.get("image_quality") or {})
    prediction = dict(result.get("prediction") or {})
    uncertainty = dict(result.get("uncertainty") or {})
    model = dict(result.get("model") or {})
    outputs_in = dict(result.get("outputs") or {})
    performance = dict(result.get("performance") or {})
    threshold_info = dict(result.get("thresholds") or {})
    model_provenance = dict(result.get("model_provenance") or {})

    run_id = metadata.get("run_id") or f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    generated_at = metadata.get("generated_at") or datetime.now().isoformat(timespec="seconds")
    quality_thresholds = dict(result.get("quality_thresholds") or threshold_info.get("quality") or quality.get("thresholds") or {})
    uncertainty_thresholds = dict(result.get("uncertainty_thresholds") or threshold_info.get("uncertainty") or uncertainty.get("thresholds") or {})
    probabilities = _coerce_probabilities(prediction.get("probabilities"))
    probability_rows = _probability_rows(probabilities)
    ranked = sorted(probability_rows, key=lambda row: row["rank"])
    top_row = ranked[0] if ranked else None
    second_row = ranked[1] if len(ranked) > 1 else None
    top_two_difference = None if not top_row or not second_row else top_row["probability"] - second_row["probability"]

    class_id = prediction.get("class_id")
    if isinstance(class_id, str) and class_id.isdigit():
        class_id = int(class_id)
    class_id = class_id if isinstance(class_id, int) and class_id in DR_SCREENING_LABELS else None
    if top_row and class_id is None:
        class_id = int(top_row["class_id"])
    class_name = DR_SCREENING_LABELS.get(class_id, prediction.get("class_name") or "Not evaluated")
    quality_status = str(quality.get("status") or "unknown")
    uncertainty_reasons = {item for item in str(uncertainty.get("reason") or "").split(",") if item}
    uncertainty_threshold_failed = bool({"low_confidence", "high_entropy", "low_top2_margin"} & uncertainty_reasons)
    fallback_used = bool(model.get("fallback_mode") or model_provenance.get("fallback_used"))
    if fallback_used:
        uncertainty["manual_review"] = True
        uncertainty["reason"] = _append_reason(str(uncertainty.get("reason") or ""), "fallback_model_used")

    prediction_for_rules = {**prediction, "class_id": class_id}
    valid_disposition = _is_disposition_valid(quality_status, prediction_for_rules, probabilities, uncertainty, fallback_used)
    referral = _derive_referral(quality_status, prediction_for_rules, uncertainty, fallback_used, model.get("reason"))
    manual_review = bool(referral["manual_review_required"] or uncertainty.get("manual_review"))
    endpoints = _clinical_endpoints(valid_disposition, class_id)

    patient = {
        "patient_id": _clean_text(patient_in.get("patient_id") or metadata.get("patient_id"), "Not provided"),
        "age": _clean_text(patient_in.get("age"), "Not provided"),
        "sex": _clean_text(patient_in.get("sex"), "Not provided"),
        "diabetes_type": _clean_text(patient_in.get("diabetes_type"), "Not provided"),
        "known_duration_of_diabetes": _clean_text(patient_in.get("known_duration_of_diabetes"), "Not provided"),
        "latest_hba1c": _clean_text(patient_in.get("latest_hba1c"), "Not provided"),
        "blood_pressure": _clean_text(patient_in.get("blood_pressure"), "Not provided"),
        "previous_dr_history": _clean_text(patient_in.get("previous_dr_history"), "Not provided"),
        "current_visual_symptoms": _clean_text(patient_in.get("current_visual_symptoms"), "Not provided"),
    }
    acquisition = {
        "eye_laterality": _clean_text(acquisition_in.get("eye_laterality"), "Unknown").lower(),
        "capture_device": _clean_text(acquisition_in.get("capture_device"), "Not provided"),
        "screening_site": _clean_text(acquisition_in.get("screening_site") or metadata.get("screening_site") or metadata.get("site_id"), "Not provided"),
        "operator_id": _clean_text(acquisition_in.get("operator_id"), "Not provided"),
    }
    model_path = model.get("active_path") or model.get("requested_path")
    threshold_path = threshold_info.get("path") or model.get("thresholds_path")
    model_metadata = dict(model.get("metadata") or {})

    return {
        "report_version": REPORT_VERSION,
        "metadata": {
            "report_id": run_id,
            "run_id": run_id,
            "generated_at": generated_at,
            "report_version": REPORT_VERSION,
            "software_report_version": SOFTWARE_REPORT_VERSION,
            "site_id": _clean_text(metadata.get("site_id"), "Not provided"),
        },
        "patient": patient,
        "acquisition": acquisition,
        "preprocessing": {**dict(result.get("preprocessing") or {}), "image_size": (result.get("preprocessing") or {}).get("image_size", "Not evaluated")},
        "image_quality": {
            "status": quality_status,
            "gradability": "Gradable" if quality_status == "accepted" else "Ungradable",
            "metrics": _quality_metrics(quality, quality_thresholds),
            "reasons": list(quality.get("reasons") or []),
            "overall_interpretation": "Image quality acceptable for the screening workflow." if quality_status == "accepted" else "Image quality rejected; repeat acquisition or manual assessment is required.",
            "thresholds": quality_thresholds,
        },
        "dr_grading": {
            "status": prediction.get("status") or "not_run",
            "predicted_class_id": class_id,
            "predicted_class_name": class_name,
            "predicted_screening_class": class_name,
            "screening_classification": "AI screening classification, not a confirmed diagnosis.",
            "confidence": top_row["probability"] if top_row else prediction.get("confidence"),
            "confidence_category": _confidence_category(top_row["probability"] if top_row else prediction.get("confidence"), uncertainty_thresholds),
            "probabilities": probability_rows,
            "highest_probability_class": top_row,
            "second_highest_class": second_row,
            "top_two_probability_difference": top_two_difference,
            "usable_for_disposition": valid_disposition,
        },
        "clinical_endpoints": endpoints,
        "lesion_analysis": {row: {"status": "Not evaluated by the current model.", "value": None, "source": "No validated lesion-level detector is currently integrated."} for row in LESION_ROWS},
        "uncertainty": {
            **uncertainty,
            "manual_review": manual_review,
            "threshold_manual_review": uncertainty_threshold_failed,
            "confidence": top_row["probability"] if top_row else uncertainty.get("confidence"),
            "confidence_threshold": uncertainty_thresholds.get("min_confidence"),
            "entropy": uncertainty.get("entropy"),
            "entropy_threshold": uncertainty_thresholds.get("max_entropy"),
            "margin": uncertainty.get("margin"),
            "margin_threshold": uncertainty_thresholds.get("min_top2_margin"),
            "why_manual_review": referral["rationale"] if manual_review else "No manual-review threshold was triggered.",
            "thresholds": uncertainty_thresholds,
        },
        "referral": referral,
        "model_provenance": {
            "model_name": _clean_text(model_metadata.get("model_name") or model.get("name"), "Not available"),
            "model_type": _clean_text(model.get("kind"), "Not available"),
            "model_version": _clean_text(model_metadata.get("version") or model_metadata.get("arch"), "Not provided"),
            "active_artifact": _safe_artifact_id(model_path),
            "checkpoint_sha256": model_provenance.get("checkpoint_sha256") or _sha256(model_path) or "Not available",
            "primary_model_available": bool(model_path and Path(model_path).exists() and not fallback_used),
            "fallback_used": fallback_used,
            "threshold_configuration_version": _clean_text(threshold_info.get("version"), "Not provided"),
            "threshold_file_sha256": model_provenance.get("threshold_sha256") or _sha256(threshold_path) or "Not available",
            "software_report_version": SOFTWARE_REPORT_VERSION,
            "input_image_sha256": _sha256(image_path) or "Not available",
            "preprocessing_size": _clean_text((result.get("preprocessing") or {}).get("image_size"), "Not evaluated"),
            "screening_site": acquisition["screening_site"],
            "generation_timestamp": generated_at,
        },
        "performance": {"inference_latency_ms": performance.get("latency_ms"), **performance},
        "clinician_review": {"clinician_decision": "", "final_clinician_grade": "", "referral_decision": "", "notes": "", "clinician_name": "", "registration_number": "", "signature": "", "review_date": ""},
        "outputs": {
            "report_file": _safe_artifact_id(outputs_in.get("report_path")),
            "json_sidecar_file": _safe_artifact_id(outputs_in.get("json_path")),
            "gradcam_file": _safe_artifact_id(outputs_in.get("gradcam_path") or outputs_in.get("explanation_path")),
            "processed_image_file": _safe_artifact_id(outputs_in.get("processed_image_path")),
            "gradcam_status": _clean_text(outputs_in.get("gradcam_status"), "Not available"),
            "gradcam_message": _clean_text(outputs_in.get("gradcam_message"), "Grad-CAM status not provided"),
        },
    }


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=NAVY),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=NAVY, spaceBefore=8, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=BLUE, spaceBefore=6, spaceAfter=6),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9, leading=12, textColor=TEXT),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontSize=8, leading=10, textColor=GREY),
        "table": ParagraphStyle("table", parent=base["BodyText"], fontSize=8, leading=10, textColor=TEXT),
        "table_bold": ParagraphStyle("table_bold", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=8, leading=10, textColor=TEXT),
        "warning": ParagraphStyle("warning", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=colors.white),
    }


def _status_color(name: str) -> colors.Color:
    return {"green": GREEN, "amber": AMBER, "red": RED, "grey": GREY, "teal": TEAL}.get(name, GREY)


def _banner(text: str, styles: dict[str, ParagraphStyle], color: colors.Color, width: float) -> Table:
    table = Table([[_p(text, styles["warning"], limit=700)]], colWidths=[width], hAlign="LEFT")
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color), ("BOX", (0, 0), (-1, -1), 0.35, color), ("PADDING", (0, 0), (-1, -1), 8)]))
    return table


def _table(data: list[list[Any]], col_widths: list[float], header: bool = True) -> Table:
    table = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1 if header else 0)
    style = [
        ("GRID", (0, 0), (-1, -1), 0.35, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        style.extend([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white)])
    table.setStyle(TableStyle(style))
    return table


def _cards(items: list[tuple[str, str, str]], styles: dict[str, ParagraphStyle], width: float) -> Table:
    cols = 4
    rows = [[[_p(label, styles["small"], limit=80), _p(value, styles["table_bold"], limit=140)] for label, value, _color in items[index : index + cols]] for index in range(0, len(items), cols)]
    table = Table(rows, colWidths=[width / cols] * cols, hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.35, BORDER), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 7)]
    for index, item in enumerate(items):
        style.append(("LINEBEFORE", (index % cols, index // cols), (index % cols, index // cols), 4, _status_color(item[2])))
    table.setStyle(TableStyle(style))
    return table


def _bar(probability: float, is_top: bool, width: float = 120, height: float = 9) -> Drawing:
    drawing = Drawing(width, height + 2)
    drawing.add(Rect(0, 1, width, height, fillColor=LIGHT_GREY, strokeColor=BORDER, strokeWidth=0.25))
    drawing.add(Rect(0, 1, max(0, min(width, width * probability)), height, fillColor=TEAL if is_top else BLUE, strokeColor=None))
    return drawing


def _image_block(path: str | Path | None, label: str, max_width: float, max_height: float, styles: dict[str, ParagraphStyle]) -> list[Any]:
    if not path or not Path(path).exists():
        return [_p(label, styles["table_bold"]), _p("Not available", styles["small"])]
    try:
        reader = ImageReader(str(path))
        image_width, image_height = reader.getSize()
        scale = min(max_width / image_width, max_height / image_height)
        return [_p(label, styles["table_bold"]), Image(str(path), width=image_width * scale, height=image_height * scale)]
    except Exception:
        return [_p(label, styles["table_bold"]), _p("Image could not be embedded.", styles["small"])]


def _section(title: str, styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [Paragraph(escape(title), styles["h1"])]


def _field_table(rows: list[tuple[str, Any]], styles: dict[str, ParagraphStyle], widths: list[float]) -> Table:
    data = [[_p("Field", styles["table_bold"]), _p("Value", styles["table_bold"])]]
    data.extend([[_p(label, styles["table_bold"]), _p(value, styles["table"], limit=700)] for label, value in rows])
    return _table(data, widths)



def _pair_table(rows: list[tuple[str, Any]], styles: dict[str, ParagraphStyle], width: float) -> Table:
    data = [[_p("Metric", styles["table_bold"]), _p("Value", styles["table_bold"]), _p("Metric", styles["table_bold"]), _p("Value", styles["table_bold"])]]
    for index in range(0, len(rows), 2):
        left = rows[index]
        right = rows[index + 1] if index + 1 < len(rows) else ("", "")
        data.append([_p(left[0], styles["table_bold"]), _p(left[1], styles["table"], limit=500), _p(right[0], styles["table_bold"], default=""), _p(right[1], styles["table"], default="", limit=500)])
    return _table(data, [width * 0.22, width * 0.25, width * 0.22, width * 0.25])

def _header_footer(report_data: dict[str, Any]):
    run_id = report_data["metadata"]["run_id"]

    def draw(canvas, doc):
        canvas.saveState()
        width, height = A4
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica-Bold", 9)
        canvas.drawString(doc.leftMargin, height - 15 * mm, "RetinaAI")
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY)
        canvas.drawRightString(width - doc.rightMargin, height - 15 * mm, f"Run ID: {_clean_text(run_id, limit=80)}")
        canvas.setStrokeColor(BORDER)
        canvas.line(doc.leftMargin, height - 18 * mm, width - doc.rightMargin, height - 18 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawString(doc.leftMargin, 10 * mm, "Screening result only - clinician review required before care decisions.")
        canvas.drawRightString(width - doc.rightMargin, 10 * mm, f"Page {doc.page}")
        canvas.restoreState()

    return draw


def _summary_section(report_data: dict[str, Any], styles: dict[str, ParagraphStyle], width: float) -> list[Any]:
    quality = report_data["image_quality"]
    grading = report_data["dr_grading"]
    endpoints = report_data["clinical_endpoints"]
    uncertainty = report_data["uncertainty"]
    referral = report_data["referral"]
    referable = endpoints["referable_dr"]["status"].title()
    sight = endpoints["sight_threatening_dr"]["status"].title()
    return [
        *_section("Screening Summary", styles),
        _cards(
            [
                ("Image quality", f"{quality['status']} / {quality['gradability']}", "green" if quality["status"] == "accepted" else "red"),
                ("Predicted DR screening class", grading["predicted_screening_class"], "teal" if grading["usable_for_disposition"] else "grey"),
                ("Referable DR", referable, "red" if referable == "Positive" else "amber" if referable == "Indeterminate" else "grey"),
                ("Sight-threatening DR suspicion", sight, "red" if sight == "Positive" else "amber" if sight == "Indeterminate" else "grey"),
                ("Confidence category", grading["confidence_category"], "green" if grading["confidence_category"] == "High" else "amber" if grading["confidence_category"] in {"Moderate", "Low"} else "grey"),
                ("Manual review", "Required" if uncertainty["manual_review"] else "Not required", "amber" if uncertainty["manual_review"] else "green"),
                ("Referral category", referral["category"], referral["color"]),
            ],
            styles,
            width,
        ),
        Spacer(1, 8),
    ]


def _probability_section(report_data: dict[str, Any], styles: dict[str, ParagraphStyle], width: float) -> list[Any]:
    grading = report_data["dr_grading"]
    rows = grading["probabilities"]
    story: list[Any] = [*_section("Prediction Breakdown", styles)]
    if not rows:
        story.append(_p("Class probabilities were unavailable or malformed, so the prediction breakdown is not evaluated.", styles["body"]))
        return story
    data = [[_p("Class", styles["table_bold"]), _p("Probability", styles["table_bold"]), _p("Rank", styles["table_bold"]), _p("Bar", styles["table_bold"])]]
    for row in rows:
        style = styles["table_bold"] if row["is_top"] else styles["table"]
        suffix = " (highest)" if row["is_top"] else " (second highest)" if row["is_second"] else ""
        data.append([_p(row["class_name"] + suffix, style), _p(_pct(row["probability"]), style), _p(str(row["rank"]), style), _bar(row["probability"], row["is_top"], width=width * 0.28)])
    table = _table(data, [width * 0.34, width * 0.16, width * 0.12, width * 0.30])
    for index, row in enumerate(rows, start=1):
        if row["is_top"]:
            table.setStyle(TableStyle([("BACKGROUND", (0, index), (-1, index), colors.HexColor("#E0F2FE"))]))
    top = grading.get("highest_probability_class") or {}
    second = grading.get("second_highest_class") or {}
    uncertainty = report_data["uncertainty"]
    details = [
        ("Highest-probability class", top.get("class_name", "Not evaluated")),
        ("Second-highest class", second.get("class_name", "Not evaluated")),
        ("Top-two probability difference", _pct(grading.get("top_two_probability_difference"))),
        ("Model confidence", _pct(grading.get("confidence"))),
        ("Predictive entropy", _fmt(uncertainty.get("entropy"))),
        ("Top-two margin", _fmt(uncertainty.get("margin"))),
        ("Confidence threshold", _fmt(uncertainty.get("confidence_threshold"))),
        ("Entropy threshold", _fmt(uncertainty.get("entropy_threshold"))),
        ("Margin threshold", _fmt(uncertainty.get("margin_threshold"))),
        ("Manual-review rationale", uncertainty.get("why_manual_review")),
    ]
    return story + [table, Spacer(1, 8), _pair_table(details, styles, width), _p("Raw softmax probabilities are model outputs for screening workflow routing and are not diagnostic certainty.", styles["small"])]


def _quality_section(report_data: dict[str, Any], styles: dict[str, ParagraphStyle], width: float) -> list[Any]:
    quality = report_data["image_quality"]
    rows = [[_p("Metric", styles["table_bold"]), _p("Value", styles["table_bold"]), _p("Threshold", styles["table_bold"]), _p("Status", styles["table_bold"]), _p("Interpretation", styles["table_bold"])]]
    for metric in quality["metrics"]:
        rows.append([_p(metric["metric"], styles["table"]), _p(metric["value"], styles["table"]), _p(metric["threshold"], styles["table"]), _p(metric["status"], styles["table_bold"]), _p(metric["interpretation"], styles["table"])])
    additional = [[_p("Additional image check", styles["table_bold"]), _p("Status", styles["table_bold"]), _p("Reason", styles["table_bold"])]]
    for check in ADDITIONAL_IMAGE_CHECKS:
        additional.append([_p(check, styles["table"]), _p("Not evaluated", styles["table_bold"]), _p("No validated component currently measures this item.", styles["table"])])
    reasons = quality["reasons"] or ["none"]
    return [
        *_section("Image-Quality Assessment", styles),
        _p(quality["overall_interpretation"], styles["body"]),
        _table(rows, [width * 0.22, width * 0.14, width * 0.15, width * 0.12, width * 0.27]),
        Spacer(1, 7),
        _p(f"Quality-gate reasons: {', '.join(reasons)}", styles["body"]),
        Spacer(1, 8),
        Paragraph("Additional Image Checks", styles["h2"]),
        _table(additional, [width * 0.34, width * 0.18, width * 0.38]),
    ]


def _findings_section(report_data: dict[str, Any], styles: dict[str, ParagraphStyle], width: float) -> list[Any]:
    grading = report_data["dr_grading"]
    quality = report_data["image_quality"]
    uncertainty = report_data["uncertainty"]
    endpoints = report_data["clinical_endpoints"]
    referral = report_data["referral"]
    top = grading.get("highest_probability_class") or {}
    second = grading.get("second_highest_class") or {}
    bullets = [
        f"Highest-probability screening grade: {top.get('class_name', grading['predicted_screening_class'])}.",
        f"Second-highest screening grade: {second.get('class_name', 'Not evaluated')}.",
        f"Uncertainty thresholds: {'manual review triggered' if uncertainty.get('threshold_manual_review') else 'passed'} ({uncertainty.get('reason', 'not available')}).",
        f"Image quality reliability: {quality['overall_interpretation']}",
        f"Referable DR mapping: {endpoints['referable_dr']['status']}.",
        f"Manual clinical review: {'required' if referral['manual_review_required'] else 'not required by routing rules'}.",
        "Important limitation: this classifier does not perform lesion-level analysis and does not assess symptoms, treatment history, or macular involvement.",
    ]
    lesion_rows = [[_p("Lesion-level item", styles["table_bold"]), _p("Current status", styles["table_bold"])]]
    for lesion, value in report_data["lesion_analysis"].items():
        lesion_rows.append([_p(lesion, styles["table"]), _p(value["status"], styles["table_bold"])])
    story: list[Any] = [*_section("Findings and Interpretation", styles)]
    story.extend(_p(f"- {item}", styles["body"], limit=700) for item in bullets)
    story.extend([Spacer(1, 8), Paragraph("Lesion-Level Analysis", styles["h2"]), _p("No lesion-level findings are reported because the current RetinaAI classifier is not a validated lesion detector.", styles["body"]), _table(lesion_rows, [width * 0.38, width * 0.52])])
    return story


def _audit_section(report_data: dict[str, Any], styles: dict[str, ParagraphStyle], width: float) -> list[Any]:
    provenance = report_data["model_provenance"]
    performance = report_data["performance"]
    rows = [
        ("Model name", provenance["model_name"]), ("Model type", provenance["model_type"]), ("Model version", provenance["model_version"]),
        ("Active artifact", provenance["active_artifact"]), ("Model/checkpoint SHA-256", provenance["checkpoint_sha256"]), ("Fallback mode used", provenance["fallback_used"]),
        ("Threshold/configuration version", provenance["threshold_configuration_version"]), ("Threshold file SHA-256", provenance["threshold_file_sha256"]),
        ("Software/report version", provenance["software_report_version"]), ("Input image SHA-256", provenance["input_image_sha256"]),
        ("Inference latency", f"{_fmt(performance.get('inference_latency_ms'))} ms"), ("Image preprocessing size", provenance["preprocessing_size"]),
        ("Screening site", provenance["screening_site"]), ("Generation timestamp", provenance["generation_timestamp"]),
    ]
    return [*_section("Technical Audit Trail", styles), _field_table(rows, styles, [width * 0.34, width * 0.56])]


def _clinician_section(styles: dict[str, ParagraphStyle], width: float) -> list[Any]:
    rows = [("Clinician decision", "Agree / disagree / indeterminate"), ("Final clinician grade", ""), ("Referral decision", ""), ("Notes", ""), ("Clinician name", ""), ("Registration number", ""), ("Signature", ""), ("Review date", "")]
    data = [[_p("Clinician review field", styles["table_bold"]), _p("Clinician completion area", styles["table_bold"])]]
    data.extend([[_p(label, styles["table"]), _p(value or " ", styles["table"])] for label, value in rows])
    return [*_section("Clinician Review Section", styles), _p("This section is intentionally blank for qualified clinician completion. AI output and clinician conclusion must remain separate.", styles["body"]), _table(data, [width * 0.34, width * 0.56])]


def generate_report(result: dict, image_path: str | Path, output_path: str | Path, patient_id: str | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if patient_id and not (result.get("patient") or {}).get("patient_id"):
        result.setdefault("patient", {})["patient_id"] = patient_id
    report_data = build_report_data(result, image_path)
    sidecar_path = output_path.with_suffix(".json")
    report_data["outputs"]["report_file"] = output_path.name
    report_data["outputs"]["json_sidecar_file"] = sidecar_path.name

    styles = _styles()
    width = A4[0] - 2 * 18 * mm
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=24 * mm, bottomMargin=18 * mm, title="RetinaAI Screening and Referral Report")
    metadata = report_data["metadata"]
    patient = report_data["patient"]
    acquisition = report_data["acquisition"]
    quality = report_data["image_quality"]
    referral = report_data["referral"]
    provenance = report_data["model_provenance"]
    outputs = result.get("outputs", {})

    story: list[Any] = [
        Paragraph("RetinaAI", styles["title"]),
        Paragraph("AI-Assisted Retinal Screening Report", styles["h1"]),
        _banner("Screening result only - not a confirmed diagnosis. A qualified clinician must review this report before any care decision.", styles, RED, width),
        _banner(DISCLAIMER, styles, NAVY, width),
        Spacer(1, 8),
        _cards(
            [
                ("Report/run ID", metadata["run_id"], "grey"),
                ("Report version", metadata["report_version"], "grey"),
                ("Date and time", metadata["generated_at"], "grey"),
                ("Patient ID", patient["patient_id"], "grey"),
                ("Screening site", acquisition["screening_site"], "grey"),
                ("Eye laterality", acquisition["eye_laterality"], "grey"),
                ("Overall result status", referral["result"], referral["color"]),
                ("Image gradability", quality["gradability"], "green" if quality["status"] == "accepted" else "red"),
                ("Manual-review status", "Required" if referral["manual_review_required"] else "Not required", "amber" if referral["manual_review_required"] else "green"),
                ("Referral category", referral["category"], referral["color"]),
            ],
            styles,
            width,
        ),
        Spacer(1, 8),
    ]
    if provenance["fallback_used"]:
        story.extend([_banner("Primary validated model unavailable. Fallback output must not be used for automated screening disposition. Manual review is required.", styles, AMBER, width), Spacer(1, 8)])

    story.extend([
        *_section("Patient and Acquisition Information", styles),
        _field_table(
            [
                ("Patient ID", patient["patient_id"]), ("Age", patient["age"]), ("Sex", patient["sex"]), ("Eye laterality", acquisition["eye_laterality"]),
                ("Diabetes type", patient["diabetes_type"]), ("Known duration of diabetes", patient["known_duration_of_diabetes"]), ("Latest HbA1c", patient["latest_hba1c"]),
                ("Blood pressure", patient["blood_pressure"]), ("Previous diabetic-retinopathy history", patient["previous_dr_history"]), ("Current visual symptoms", patient["current_visual_symptoms"]),
                ("Capture device/camera", acquisition["capture_device"]), ("Screening site", acquisition["screening_site"]), ("Operator ID", acquisition["operator_id"]),
            ],
            styles,
            [width * 0.34, width * 0.56],
        ),
        Spacer(1, 8),
        *_summary_section(report_data, styles, width),
        PageBreak(),
        *_section("Retinal Images and Explanation", styles),
    ])

    image_table = Table(
        [[_image_block(image_path, "Original uploaded fundus image", width * 0.44, 120, styles), _image_block(outputs.get("processed_image_path"), "Processed/cropped model-input image", width * 0.44, 120, styles)]],
        colWidths=[width * 0.46, width * 0.46],
        hAlign="LEFT",
    )
    image_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("GRID", (0, 0), (-1, -1), 0.35, BORDER), ("PADDING", (0, 0), (-1, -1), 6)]))
    story.extend([
        image_table,
        Spacer(1, 10),
        Table([[_image_block(outputs.get("gradcam_path") or outputs.get("explanation_path"), "Grad-CAM overlay", width * 0.64, 125, styles)]], colWidths=[width * 0.68], hAlign="LEFT"),
        Spacer(1, 6),
        _p("Highlighted regions influenced the model output. They are not verified lesion locations and do not confirm the presence or absence of pathology.", styles["body"]),
    ])
    if report_data["outputs"]["gradcam_status"].lower() != "available":
        story.append(_p(f"Grad-CAM availability: {report_data['outputs']['gradcam_message']}", styles["small"]))
    story.extend([Spacer(1, 10), *_probability_section(report_data, styles, width), PageBreak()])
    story.extend([*_quality_section(report_data, styles, width), PageBreak(), *_findings_section(report_data, styles, width), Spacer(1, 10)])
    story.extend([
        *_section("Referral and Next Action", styles),
        _field_table(
            [
                ("Result", referral["result"]), ("Referral category", referral["category"]), ("Action", referral["action"]),
                ("Manual review required", referral["manual_review_required"]), ("Rationale", referral["rationale"]),
                ("Safety symptom note", "New sudden vision loss, flashes, floaters, eye pain or a curtain-like visual disturbance requires prompt clinical assessment regardless of this AI result."),
            ],
            styles,
            [width * 0.30, width * 0.60],
        ),
        Spacer(1, 10),
        *_clinician_section(styles, width),
        Spacer(1, 10),
        *_audit_section(report_data, styles, width),
    ])

    doc.build(story, onFirstPage=_header_footer(report_data), onLaterPages=_header_footer(report_data))
    sidecar_path.write_text(json.dumps(report_data, indent=2, allow_nan=False), encoding="utf-8")
    return output_path
