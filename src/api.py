from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Callable
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .case_store import get_case, list_cases, save_case
from .constants import DISCLAIMER
from .inference import screen_retina_image
from .models import CNN_MODEL_SPECS
from .quality_check import assess_quality, load_quality_thresholds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
UPLOAD_DIR = REPORTS_DIR / "api_uploads"
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
ROLE_RANK = {"viewer": 1, "clinician": 2, "admin": 3}

app = FastAPI(
    title="RetinaAI API",
    version="1.0.0",
    description="Production API for retinal image quality screening, diabetic retinopathy model inference, Grad-CAM artifacts, and PDF reports.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _api_keys() -> dict[str, str]:
    raw = os.getenv("RETINAAI_API_KEYS", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = dict(part.split(":", 1) for part in raw.split(",") if ":" in part)
    out: dict[str, str] = {}
    for key, value in parsed.items():
        role = value.get("role") if isinstance(value, dict) else value
        if str(role) in ROLE_RANK:
            out[str(key)] = str(role)
    return out


def principal(x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None) -> dict[str, str]:
    keys = _api_keys()
    if not keys:
        return {"id": "local-dev", "role": "admin"}
    if not x_api_key or x_api_key not in keys:
        raise HTTPException(status_code=401, detail="Valid X-API-Key header required.")
    return {"id": x_api_key[:8], "role": keys[x_api_key]}


def require_role(role: str) -> Callable:
    def _dependency(user: dict[str, str] = Depends(principal)) -> dict[str, str]:
        if ROLE_RANK[user["role"]] < ROLE_RANK[role]:
            raise HTTPException(status_code=403, detail=f"{role} role required.")
        return user

    return _dependency


def _artifact_url(path: str | None) -> str | None:
    if not path:
        return None
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(REPORTS_DIR.resolve())
    except ValueError:
        return None
    return f"/artifacts/{relative.as_posix()}"


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "image.png").suffix.lower() or ".png"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported image type.")
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    path.write_bytes(content)
    return path


def _with_urls(result: dict) -> dict:
    outputs = result.setdefault("outputs", {})
    outputs["gradcam_url"] = _artifact_url(outputs.get("gradcam_path"))
    outputs["explanation_url"] = _artifact_url(outputs.get("explanation_path"))
    outputs["report_url"] = _artifact_url(outputs.get("report_path"))
    return result


@app.get("/health", tags=["system"])
def health() -> dict:
    return {
        "status": "ok",
        "service": "retinaai-api",
        "disclaimer": DISCLAIMER,
        "reports_dir": str(REPORTS_DIR),
        "auth_configured": bool(_api_keys()),
    }


@app.post("/quality", tags=["screening"])
async def quality(
    image: UploadFile = File(...),
    thresholds: str = Query("configs/thresholds.yaml"),
    site_id: str | None = Query(None),
) -> dict:
    image_path = await _save_upload(image)
    return assess_quality(image_path, load_quality_thresholds(thresholds, site_id=site_id)).to_dict()


@app.post("/predict", tags=["screening"])
async def predict(
    image: UploadFile = File(...),
    model: str = Query("models/efficientnet_b0_torch_transfer_acc.pt"),
    fallback_model: str | None = Query("models/baseline_sklearn.pkl"),
    thresholds: str = Query("configs/thresholds.yaml"),
    patient_id: str | None = Form(None),
    site_id: str | None = Form(None),
    user: dict[str, str] = Depends(require_role("clinician")),
) -> dict:
    image_path = await _save_upload(image)
    result = screen_retina_image(
        image_path,
        model_path=model,
        fallback_model_path=fallback_model,
        thresholds_path=thresholds,
        output_dir=REPORTS_DIR / "api_reports",
        patient_id=patient_id,
        site_id=site_id,
    )
    result = _with_urls(result)
    save_case(result, principal=user)
    return result


@app.post("/gradcam", tags=["screening"])
async def gradcam(
    image: UploadFile = File(...),
    model: str = Query("models/efficientnet_b0_torch_transfer_acc.pt"),
    fallback_model: str | None = Query("models/baseline_sklearn.pkl"),
    user: dict[str, str] = Depends(require_role("clinician")),
) -> FileResponse:
    image_path = await _save_upload(image)
    result = screen_retina_image(image_path, model_path=model, fallback_model_path=fallback_model, output_dir=REPORTS_DIR / "api_reports")
    path = Path(result["outputs"]["gradcam_path"])
    return FileResponse(path, media_type="image/png", filename=path.name)


@app.post("/report", tags=["reports"])
async def report(
    image: UploadFile = File(...),
    model: str = Query("models/efficientnet_b0_torch_transfer_acc.pt"),
    fallback_model: str | None = Query("models/baseline_sklearn.pkl"),
    patient_id: str | None = Form(None),
    site_id: str | None = Form(None),
    user: dict[str, str] = Depends(require_role("clinician")),
) -> FileResponse:
    image_path = await _save_upload(image)
    result = screen_retina_image(
        image_path,
        model_path=model,
        fallback_model_path=fallback_model,
        output_dir=REPORTS_DIR / "api_reports",
        patient_id=patient_id,
        site_id=site_id,
    )
    save_case(_with_urls(result), principal=user)
    path = Path(result["outputs"]["report_path"])
    return FileResponse(path, media_type="application/pdf", filename=path.name)


@app.get("/cases", tags=["cases"])
def cases(
    limit: int = Query(50, ge=1, le=500),
    patient_id: str | None = Query(None),
    site_id: str | None = Query(None),
    user: dict[str, str] = Depends(require_role("viewer")),
) -> dict:
    rows = list_cases(limit=limit, patient_id=patient_id, site_id=site_id)
    for row in rows:
        row["report_url"] = _artifact_url(row.get("report_path"))
        row["gradcam_url"] = _artifact_url(row.get("gradcam_path"))
    return {"cases": rows}


@app.get("/cases/{run_id}", tags=["cases"])
def case_detail(run_id: str, user: dict[str, str] = Depends(require_role("viewer"))) -> dict:
    result = get_case(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Case not found.")
    return result


@app.get("/metrics", tags=["models"])
def metrics() -> dict:
    candidates = [
        REPORTS_DIR / "metrics_efficientnet_b0_torch_transfer_acc.json",
        REPORTS_DIR / "metrics_efficientnet_b0_torch_transfer.json",
        REPORTS_DIR / "comparison.json",
        REPORTS_DIR / "model_comparison.json",
        REPORTS_DIR / "metrics_baseline_sklearn.json",
    ]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {"status": "no_metrics", "message": "Run python -m src.model_comparison to generate comparison metrics."}


@app.get("/models", tags=["models"])
def models() -> dict:
    artifacts = []
    for path in (PROJECT_ROOT / "models").glob("*"):
        if path.suffix.lower() in {".pkl", ".keras", ".h5", ".pt"} or path.is_dir():
            artifacts.append(str(path.relative_to(PROJECT_ROOT)))
    return {
        "supported_cnn_models": CNN_MODEL_SPECS,
        "baseline": "baseline_sklearn",
        "artifacts": artifacts,
    }


@app.get("/artifacts/{artifact_path:path}", tags=["reports"])
def artifact(artifact_path: str, user: dict[str, str] = Depends(require_role("viewer"))) -> FileResponse:
    resolved = (REPORTS_DIR / artifact_path).resolve()
    try:
        resolved.relative_to(REPORTS_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Artifact path is not allowed.") from exc
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Artifact not found.")
    media_type = "application/pdf" if resolved.suffix.lower() == ".pdf" else "image/png"
    return FileResponse(resolved, media_type=media_type, filename=resolved.name)