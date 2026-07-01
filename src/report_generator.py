from __future__ import annotations

from datetime import datetime
from pathlib import Path
from textwrap import wrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .constants import DISCLAIMER, DR_CLASSES

PAGE_MARGIN = 42
TEXT = colors.HexColor("#111827")
MUTED = colors.HexColor("#64748B")
BORDER = colors.HexColor("#CBD5E1")
BLUE = colors.HexColor("#1D4ED8")
TEAL = colors.HexColor("#0F766E")
DANGER = colors.HexColor("#B91C1C")


def _text(c: canvas.Canvas, text: str, x: int, y: int, size: int = 10, bold: bool = False, color=TEXT) -> int:
    c.setFillColor(color)
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    c.drawString(x, y, str(text)[:140])
    return y - int(size * 1.55)


def _wrapped(c: canvas.Canvas, text: str, x: int, y: int, width_chars: int = 88, size: int = 10, color=TEXT) -> int:
    for line in wrap(str(text), width=width_chars) or [""]:
        y = _text(c, line, x, y, size=size, color=color)
    return y


def _section(c: canvas.Canvas, title: str, x: int, y: int) -> int:
    c.setStrokeColor(BORDER)
    c.line(x, y + 7, 553, y + 7)
    return _text(c, title, x, y, size=12, bold=True, color=BLUE) - 4


def _box(c: canvas.Canvas, x: int, y: int, w: int, h: int, label: str, value: str, accent=BLUE) -> None:
    c.setStrokeColor(BORDER)
    c.setFillColor(colors.white)
    c.roundRect(x, y - h, w, h, 6, stroke=1, fill=1)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(x + 10, y - 17, label[:34])
    c.setFillColor(accent)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x + 10, y - 38, str(value)[:32])


def _draw_explanation(c: canvas.Canvas, explanation_path: str | None, x: int, y: int) -> int:
    if not explanation_path or not Path(explanation_path).exists():
        return _text(c, "Grad-CAM image: not available", x, y, color=MUTED)
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(BLUE)
    c.drawString(x, y, "Grad-CAM / Explanation")
    y -= 180
    try:
        c.drawImage(ImageReader(explanation_path), x, y, width=252, height=160, preserveAspectRatio=True, anchor="nw")
    except Exception:
        y += 180
        return _text(c, "Explanation image could not be embedded.", x, y - 20, color=MUTED)
    return y - 18


def _fmt(value, default: str = "not available") -> str:
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def generate_report(result: dict, image_path: str | Path, output_path: str | Path, patient_id: str | None = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = result.get("metadata", {})
    patient_id = patient_id or metadata.get("patient_id") or "Unspecified"
    generated_at = metadata.get("generated_at") or datetime.now().isoformat(timespec="seconds")
    quality = result.get("quality", {})
    prediction = result.get("prediction", {})
    uncertainty = result.get("uncertainty", {})
    outputs = result.get("outputs", {})
    recommendation = result.get("recommendation", {})

    c = canvas.Canvas(str(output_path), pagesize=A4)
    c.setTitle("RetinaAI Screening Report")
    width, height = A4
    y = int(height) - PAGE_MARGIN

    c.setFillColor(BLUE)
    c.rect(0, int(height) - 88, int(width), 88, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(PAGE_MARGIN, int(height) - 42, "RetinaAI Screening Report")
    c.setFont("Helvetica", 9)
    c.drawString(PAGE_MARGIN, int(height) - 62, "Clinical review support - not a diagnostic medical device")
    c.setFont("Helvetica-Bold", 10)
    c.drawRightString(int(width) - PAGE_MARGIN, int(height) - 42, f"Patient ID: {patient_id}")
    c.setFont("Helvetica", 9)
    c.drawRightString(int(width) - PAGE_MARGIN, int(height) - 62, f"Date: {generated_at}")
    y -= 78

    status_color = TEAL if quality.get("status") == "accepted" else DANGER
    class_id = prediction.get("class_id")
    class_name = prediction.get("class_name") or (DR_CLASSES.get(class_id) if class_id is not None else "not available")
    _box(c, PAGE_MARGIN, y, 160, 56, "Quality", quality.get("status", "unknown"), status_color)
    _box(c, PAGE_MARGIN + 176, y, 160, 56, "Prediction", class_name, BLUE)
    _box(c, PAGE_MARGIN + 352, y, 160, 56, "Confidence", _fmt(prediction.get("confidence")), TEAL)
    y -= 78

    y = _section(c, "Quality Metrics", PAGE_MARGIN, y)
    y = _text(c, f"Blur: {_fmt(quality.get('blur_score'))} | Brightness: {_fmt(quality.get('brightness_score'))} | Contrast: {_fmt(quality.get('contrast_score'))} | Retina visibility: {_fmt(quality.get('retina_visibility_score'))}", PAGE_MARGIN, y)
    y = _text(c, f"Quality gate reasons: {', '.join(quality.get('reasons') or ['none'])}", PAGE_MARGIN, y, color=MUTED)
    y -= 8

    y = _section(c, "Prediction and Uncertainty", PAGE_MARGIN, y)
    y = _text(c, f"Prediction status: {prediction.get('status', 'unknown')}", PAGE_MARGIN, y)
    y = _text(c, f"Severity class: {class_name}", PAGE_MARGIN, y)
    y = _text(c, f"Predictive entropy: {_fmt(uncertainty.get('entropy'))} | Top-2 margin: {_fmt(uncertainty.get('margin'))}", PAGE_MARGIN, y)
    y = _text(c, f"Manual review required: {uncertainty.get('manual_review')}", PAGE_MARGIN, y, bold=True, color=DANGER if uncertainty.get("manual_review") else TEAL)
    y = _text(c, f"Routing reason: {uncertainty.get('reason', 'not available')}", PAGE_MARGIN, y, color=MUTED)
    probabilities = prediction.get("probabilities")
    if probabilities:
        y = _wrapped(c, f"Class probabilities: {probabilities}", PAGE_MARGIN, y, width_chars=100, color=MUTED)
    y -= 8

    y = _section(c, "Recommendation", PAGE_MARGIN, y)
    y = _wrapped(c, recommendation.get("text", "Clinical review is required before care decisions."), PAGE_MARGIN, y, width_chars=92)
    y = _text(c, f"Urgency: {recommendation.get('urgency', 'clinical review')}", PAGE_MARGIN, y, bold=True, color=BLUE)
    y -= 8

    if y < 260:
        c.showPage()
        y = int(height) - PAGE_MARGIN
    y = _draw_explanation(c, outputs.get("gradcam_path") or outputs.get("explanation_path"), PAGE_MARGIN, y)

    if y < 130:
        c.showPage()
        y = int(height) - PAGE_MARGIN
    y = _section(c, "Medical Disclaimer", PAGE_MARGIN, y)
    y = _wrapped(c, DISCLAIMER, PAGE_MARGIN, y, width_chars=96, color=TEXT)
    y -= 4
    _wrapped(c, "This report supports screening workflow triage only. A qualified clinician must review the patient, source image, model output, uncertainty values, and any additional clinical data.", PAGE_MARGIN, y, width_chars=96, size=9, color=MUTED)

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(PAGE_MARGIN, 30, f"Source image: {Path(image_path).name}")
    c.drawRightString(int(width) - PAGE_MARGIN, 30, "RetinaAI - screening prototype")
    c.save()
    return output_path