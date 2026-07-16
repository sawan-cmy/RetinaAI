# RetinaAI

RetinaAI is a high-fidelity, deep learning computer vision pipeline engineered for retinal screening workflow support, uncertainty-aware triage, and clinical decision-support research using retinal fundus photography.

> [!WARNING]
> This is a screening prototype, not a diagnostic medical device. A qualified medical professional must review every output before clinical care decisions.

## 🎯 The Core Problem & Product Utility

In traditional clinical environments within India, diagnosing sight-threatening eye conditions requires heavy, static desktop fundus cameras costing anywhere from ₹12 Lakhs to ₹75 Lakhs+. Because of this massive financial barrier, rural clinics, tier-3 cities, and community medical camps cannot afford proper screening equipment. Furthermore, reading these scans demands an on-call, highly specialized ophthalmologist—creating a massive logistical bottleneck.

RetinaAI bridges this economic and infrastructural gap:

- **Real-Time Clinical Triaging**: Instead of forcing hundreds of routine screening scans to sit in an unread queue for days, the system acts as a real-time digital assistant. It screens images, checks quality, and flags outputs that need manual review or referral workflow attention so clinicians can prioritize review queues.
- **Democratizing Public Healthcare**: By keeping the software entirely hardware-agnostic, medical operations can pair RetinaAI with highly portable, mass-accessible smartphone lenses. A field nurse in a remote village can use an ultra-low-cost clip-on lens like the D-Eye (₹30,000 - ₹35,000) or a clinical hand-held phone system like the Remidio FOP (₹3.5 Lakhs - ₹4.2 Lakhs) to snap a high-res photo on an ordinary phone, upload it to our pipeline, and receive an instant screening summary for clinician review.
- **Objective Quantification**: Supports consistent screening workflow review by outputting a stable model probability matrix with uncertainty routing.

## ⚙️ Technical Capabilities

- **Automated Image Quality Gating & Preprocessing**:
  - Automatically assesses input fundus photos to reject blurry, low-contrast, or artifact-heavy uploads using luminance and gradient-based checks ([quality_check.py](file:///c:/Users/sawan/Desktop/new_project/RetinaAI/src/quality_check.py)).
  - Automated black-border cropping and aspect-ratio-preserving resizing to normalize clinical inputs from varied camera types.
- **Multi-Model Deep Learning Pipeline**:
  - Leverages advanced Transfer Learning CNN architectures: **EfficientNet-B0**, **EfficientNet-B3**, and **ResNet50** for robust diabetic retinopathy grading.
  - Multi-dataset loader abstraction supporting diverse standard formats (**APTOS 2019**, **EyePACS**, **Messidor**, and **IDRiD**-style label schemas).
  - Built-in fail-safe fallback: Automatically routes predictions to a baseline **scikit-learn Random Forest** classifier if deep learning checkpoints are missing or invalid, preventing system crashes during runtime.
- **Explainable AI (XAI) & Model Explanation**:
  - Computes and renders **Grad-CAM (Gradient-weighted Class Activation Mapping)** overlays for CNN artifacts. Highlighted regions influenced model output; they are not verified lesion locations.
  - Clean explanation fallbacks showing descriptive "Not Available" visual overlays when executing the non-convolutional baseline models.
- **Uncertainty-Aware Routing & Clinical Triaging**:
  - Analyzes prediction confidence, predictive entropy, and top-2 class margin.
  - Flags and routes highly uncertain predictions or failed quality checks for manual ophthalmologist review.
- **Full-Stack Diagnostics Dashboard & Reports**:
  - Professional Next.js (TypeScript) web-dashboard for real-time patient queue viewing, uploading, Grad-CAM visualization, and historical analytics.
  - Automated PDF screening report generator with optional metadata, quality metrics, probability breakdown, Grad-CAM limitations, deterministic referral routing, clinician-review fields, audit trail, and standard safety disclaimers.
- **Production-Ready & Secure Core**:
  - FastAPI-powered backend exposing high-performance endpoints with role-based API-key authentication (`viewer`, `clinician`, `admin`).
  - Persistent SQLite databases for case history tracking.
  - Fully containerized ecosystem using Docker and Docker Compose for seamless localized or cloud deployment.


## Architecture

```text
Retinal image upload
  -> preprocessing.crop_black_borders / resize
  -> quality_check.assess_quality
  -> CNN prediction: EfficientNet-B0, EfficientNet-B3, or ResNet50
  -> fallback baseline: sklearn RandomForest if CNN artifact is missing
  -> uncertainty: confidence, predictive entropy, top-2 margin
  -> Grad-CAM overlay for CNNs, explicit unavailable image for fallback
  -> recommendation and manual-review routing
  -> PDF report
  -> FastAPI + Next.js dashboard
```

## Folder Structure

```text
app/                    Streamlit compatibility app
configs/                Training and threshold configuration
data/                   Raw/external dataset placeholders
src/                    ML, inference, API, report, and validation modules
frontend/               Next.js TypeScript dashboard
tests/                  Pytest coverage and self-checks
reports/                Generated metrics, figures, PDFs
models/                 Local model artifacts, ignored by default
.github/workflows/      CI
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
```

Optional CNN training stack:

```powershell
pip install -r requirements-deep-learning.txt
```

Optional PDF inspection stack:

```powershell
pip install pdfplumber pypdf
# Install Poppler separately so `pdftoppm` is available for visual PDF rendering.
# If Poppler is unavailable, `pip install pypdfium2` can render local QA PNGs.
```

Frontend:

```powershell
cd frontend
npm install
```

## Dataset Setup

Supported dataset names: `aptos`, `aptos2019`, `eyepacs`, `messidor`, `idrid`.

The loader is configurable and expects a CSV with an image id column and label column. Built-in column candidates cover common layouts, and reusable label mappings live under `configs/label_mappings/`.

APTOS example:

```text
data/raw/aptos2019/
|-- train.csv
`-- images_288_scaled/
```

Download helper:

```powershell
.\scripts\download_training_data.ps1
```

## Training

Baseline random forest:

```powershell
python -m src.train --model baseline_sklearn --labels-csv data/raw/aptos2019/train.csv --image-dir data/raw/aptos2019/images_288_scaled
```

Transfer learning:

```powershell
python -m src.train --model efficientnet_b0 --labels-csv data/raw/aptos2019/train.csv --image-dir data/raw/aptos2019/images_288_scaled
python -m src.train --model efficientnet_b3 --labels-csv data/raw/aptos2019/train.csv --image-dir data/raw/aptos2019/images_288_scaled
python -m src.train --model resnet50 --labels-csv data/raw/aptos2019/train.csv --image-dir data/raw/aptos2019/images_288_scaled
```

CNN training supports checkpoints, early stopping, mixed precision, learning-rate reduction, class weighting, and reproducible seeds through `configs/train.yaml`. Each completed training run writes metrics, calibration curves, tuned site thresholds, a model card, a dataset card, and a ZIP metrics package.

## GPU Notes

An RTX 3050 can train these models, but TensorFlow 2.21 on native Windows does not use CUDA GPUs. Use WSL2 with NVIDIA CUDA support, or a compatible DirectML setup, before expecting GPU acceleration.

## Evaluation

Model comparison:

```powershell
python -m src.model_comparison --labels-csv data/raw/aptos2019/train.csv --image-dir data/raw/aptos2019/images_288_scaled --models baseline_sklearn,efficientnet_b0,efficientnet_b3,resnet50
```

Outputs:

- `reports/comparison.csv`
- `reports/comparison.json`
- `reports/comparison.png`
- `reports/figures/model_comparison.png`
- `reports/figures/confusion_matrix_best_model.png`

External validation with documented mappings:

```powershell
python -m src.external_validation --dataset eyepacs --model models/efficientnet_b0.keras --labels-csv data/external/eyepacs/labels.csv --image-dir data/external/eyepacs/images --label-map configs/label_mappings/eyepacs_5class.json --site eyepacs_site
python -m src.external_validation --dataset messidor --model models/efficientnet_b0.keras --labels-csv data/external/messidor/labels.csv --image-dir data/external/messidor/images --label-map configs/label_mappings/messidor_retinopathy_grade_0_3.json --site messidor_site
python -m src.external_validation --dataset idrid --model models/efficientnet_b0.keras --labels-csv data/external/idrid/labels.csv --image-dir data/external/idrid/images --label-map configs/label_mappings/idrid_5class.json --site idrid_site
```

External validation writes separate metrics, prediction rows, calibration curves, and tuned thresholds under `reports/external_validation/`.

Calibration and packaging can also be run directly:

```powershell
python -m src.calibration --predictions reports/external_validation/eyepacs_predictions.csv --site eyepacs_site --model efficientnet_b0
python -m src.artifact_package --model efficientnet_b0
```

## Inference

CLI:

```powershell
python -m src.inference --image tests/_self_check/synthetic_retina.png --model models/efficientnet_b0_torch_transfer_acc.pt --fallback-model models/baseline_sklearn.pkl --site-id aptos_internal
```

If the CNN is missing, inference does not crash. It falls back to the random-forest baseline when available, routes uncertain cases to manual review, creates an explicit Grad-CAM-unavailable image, and still generates a PDF report.

## Screening Report v2

`src.report_generator.generate_report` now creates a 3-5 page **RetinaAI Screening and Referral Report** plus a machine-readable JSON sidecar next to each PDF. The report is generated from values already produced by the system or entered by the user; it does not use an LLM and does not invent lesions, symptoms, history, treatment, or examination findings.

Report contents:

- Header with report/run ID, report version, timestamp, patient ID, screening site, eye laterality, image gradability, manual-review status, referral category, and prominent screening-only disclaimer.
- Optional patient/acquisition metadata with `Not provided` for missing fields.
- Screening summary cards for image quality, AI screening class, referable DR, sight-threatening DR suspicion, confidence category, manual review, and referral category.
- Original image, processed model-input image when available, and Grad-CAM overlay when available. Grad-CAM is documented as model-output influence only, not lesion detection.
- Five-class probability bar chart/table for No apparent DR, Mild NPDR, Moderate NPDR, Severe NPDR, and Proliferative DR, with rank, top-two difference, entropy, margin, and thresholds.
- Image-quality table using only calculated blur/sharpness, brightness, contrast, retina visibility, overall status, and quality-gate reasons. Unsupported checks such as optic-disc, macula, vessel, reflection, eyelash, and field-of-view assessment are explicitly `Not evaluated`.
- Deterministic findings, referral/next-action section, clinician-review section, lesion-level analysis rows marked `Not evaluated by the current model`, and technical audit trail with safe artifact identifiers and SHA-256 hashes where available.

Safety rules:

- The report is a screening and clinical decision-support prototype, not a confirmed diagnosis.
- Fallback Random Forest output is marked not validated for automated disposition and always requires manual review.
- Poor-quality, missing-model, model-error, fallback, malformed-probability, and high-uncertainty cases are conservative and indeterminate.
- Recommendations are deterministic and rule-based. They do not recommend medication, injections, laser treatment, surgery, or exact follow-up intervals.
- Lesion analysis is not currently available. Microaneurysms, retinal haemorrhages, hard exudates, soft exudates, neovascularization, and macular involvement are all shown as not evaluated.

Optional metadata fields accepted by the CLI/API/frontend upload flow:

```text
patient_id, age, sex, eye_laterality, diabetes_type, known_duration_of_diabetes,
latest_hba1c, blood_pressure, previous_dr_history, current_visual_symptoms,
capture_device, screening_site, operator_id
```

CLI example:

```powershell
python -m src.inference --image tests/_self_check/synthetic_retina.png --patient-id TEST-001 --eye-laterality left --screening-site qa
```

API example:

```powershell
curl -X POST http://127.0.0.1:8000/predict `
  -F "image=@tests/_self_check/synthetic_retina.png" `
  -F "patient_id=TEST-001" `
  -F "eye_laterality=left" `
  -F "screening_site=qa"
```

Structured sidecar schema:

```json
{
  "report_version": "2.0",
  "metadata": {},
  "patient": {},
  "acquisition": {},
  "preprocessing": {},
  "image_quality": {},
  "dr_grading": {},
  "clinical_endpoints": {
    "any_dr": {},
    "referable_dr": {},
    "sight_threatening_dr": {}
  },
  "lesion_analysis": {},
  "uncertainty": {},
  "referral": {},
  "model_provenance": {},
  "performance": {},
  "clinician_review": {},
  "outputs": {}
}
```

Rendered synthetic sample pages for layout QA are written under `reports/pdf_render_check_v2/rendered_pdfium/` when PDFium is available locally. Do not use real patient information or identifiable retinal images for samples.

## API

Start FastAPI:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `POST /predict`
- `POST /quality`
- `POST /gradcam`
- `POST /report`
- `GET /cases`
- `GET /cases/{run_id}`
- `GET /metrics`
- `GET /models`

OpenAPI docs: `http://127.0.0.1:8000/docs`.

Case history is stored server-side in `reports/case_history.sqlite3`. Set `RETINAAI_API_KEYS` to enable API-key roles, for example `{"viewer-key":"viewer","clinician-key":"clinician","admin-key":"admin"}`. When unset, local development runs as `admin`.

## Frontend

```powershell
cd frontend
$env:RETINAAI_API_URL="http://127.0.0.1:8000"
npm run dev -- --hostname 127.0.0.1 --port 3000
```

Pages: Dashboard, Upload, Prediction, Grad-CAM Viewer, Metrics, Reports, Settings, Model Comparison, History.

## Vercel Preview

The frontend can be deployed to Vercel from `frontend/`. The preview build is useful for sharing dashboard, metrics, reports, and history screens. Upload/demo screening requires `RETINAAI_API_URL` to point at a deployed FastAPI backend.

## Docker

Backend only:

```powershell
docker build -t retinaai-api .
docker run --rm -p 8000:8000 retinaai-api
```

Full stack:

```powershell
docker compose up --build
```

Frontend: `http://127.0.0.1:3000`  API: `http://127.0.0.1:8000`.

## Testing

```powershell
python tests/self_check.py
python -m pytest
python -m compileall -q app src tests
cd frontend
npm run lint
npm run typecheck
npm run build
npm run test:e2e
```

## Screenshots

![Upload Screen](docs/screenshots/upload.png)

Add other current screenshots under `docs/screenshots/` after running the app:

- `docs/screenshots/dashboard.png`
- `docs/screenshots/gradcam.png`
- `docs/screenshots/report.png`

## Results

The current local primary model is `models/efficientnet_b0_torch_transfer_acc.pt`, an EfficientNet-B0 PyTorch checkpoint with test accuracy `0.823315` in `reports/metrics_efficientnet_b0_torch_transfer_acc.json`. `reports/comparison.json` selects it over the Random Forest fallback. Do not publish metrics that were not produced by the pipeline. Training and validation artifacts are written under `reports/calibration/`, `reports/cards/`, `reports/external_validation/`, and `reports/packages/`.

## Limitations

- The local primary CNN checkpoint is `models/efficientnet_b0_torch_transfer_acc.pt`; Random Forest is only the fallback when the CNN artifact or PyTorch runtime is unavailable.
- Grad-CAM is real only for trained CNN artifacts. Baseline fallback produces an explicit unavailable explanation image.
- Site thresholds are only as good as the site validation set used to tune them.
- External validation depends on dataset access/license constraints and verified source label schemas.
- Role-based access control is API-key based for this prototype; production PHI handling still needs secure secret management, transport policy, retention policy, and audit review.

## Future Work

- Run full CNN training on the target hardware/GPU budget and archive the produced checkpoint package.
- Add production identity provider integration and encrypted case/report storage.
- Add prospective clinical validation once retrospective EyePACS, Messidor, and IDRiD runs are complete.

## Resume Bullet

Built an uncertainty-aware retinal screening platform using EfficientNet/ResNet transfer learning support, image quality gates, Grad-CAM explainability, FastAPI inference, Next.js clinical dashboard, PDF reports, model comparison, external validation hooks, calibration and threshold tuning artifacts, server-side case history, API-key RBAC, model/dataset cards, run packaging, Docker, CI, and automated tests.