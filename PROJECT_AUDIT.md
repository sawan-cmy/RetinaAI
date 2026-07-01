# Project Audit

Date: 2026-07-01

## Audit Scope

Reviewed source, configuration, docs, tests, frontend app code, package metadata, Docker setup, and generated artifact locations. Vendor/build/raw-data directories were identified but not hand-audited as source: `frontend/node_modules`, `frontend/.next`, `data/raw`, Python `__pycache__`, generated reports, and generated test outputs.

## Findings Before Fixes

- CNN model support was a stub; only the random-forest handcrafted-feature baseline was trainable.
- Dataset loading was APTOS-specific and did not support external label mapping.
- Grad-CAM could only be manually called for an externally supplied Keras model; inference always wrote unavailable placeholders.
- Inference was baseline-first and did not expose model fallback metadata, recommendation text, or latency.
- Model comparison trained only random-forest variants and did not emit the requested `comparison.*` artifacts.
- No FastAPI production backend existed.
- PDF reports were plain and missed patient ID, recommendation, uncertainty detail, and hospital-style layout.
- Frontend contained a polished but partially static dashboard, older Vite scaffold files, generated `.next` output, and fake/static model names.
- Next API route shelled out to Python; this was fragile for production builds.
- Test coverage was mostly a single self-check script.
- Docker only ran Streamlit, not the production API/frontend architecture.
- CI, pre-commit, typecheck, and Playwright test wiring were missing.
- Git is not initialized in this workspace, so file-level changed-state tracking is unavailable.

## Fixed

- Added generic dataset abstraction in `src.datasets` for APTOS, EyePACS, Messidor, and IDRiD-style CSVs with label mapping.
- Implemented TensorFlow/Keras transfer-learning model factory for EfficientNet-B0, EfficientNet-B3, and ResNet50.
- Added checkpoint metadata saving/loading for Keras `.keras` artifacts.
- Added reproducible seeds, class weighting, early stopping, mixed precision, and learning-rate scheduling in `src.train`.
- Preserved the random forest as a fallback baseline.
- Implemented real Keras Grad-CAM heatmap generation and overlay saving for CNNs.
- Upgraded inference to run quality gate, model selection, fallback, confidence, entropy, top-2 margin, Grad-CAM/unavailable artifact, recommendation, latency, and PDF report.
- Added FastAPI backend with `/predict`, `/quality`, `/gradcam`, `/report`, `/metrics`, `/models`, `/health`, and OpenAPI docs.
- Added external validation command in `src.external_validation`.
- Updated model comparison to write `reports/comparison.csv`, `reports/comparison.json`, `reports/comparison.png`, plus legacy model comparison outputs.
- Upgraded PDF reporting with patient ID, date, quality metrics, prediction, confidence, uncertainty, Grad-CAM, recommendation, disclaimer, and hospital-style formatting.
- Replaced landing-style frontend root with a professional healthcare dashboard.
- Added requested frontend pages: Dashboard, Upload, Prediction, Grad-CAM Viewer, Metrics, Reports, Settings, Model Comparison, History.
- Reworked frontend upload route to proxy to FastAPI instead of spawning Python.
- Removed active unsupported model names and decorative gradient/orb styling from the app surface.
- Added pytest coverage for preprocessing, quality, training, inference, Grad-CAM, uncertainty, and FastAPI.
- Added Playwright e2e navigation test.
- Added Dockerfile, frontend Dockerfile, Docker Compose, GitHub Actions, pre-commit config, pyproject tooling, and dev requirements.
- Rewrote README and updated architecture, dataset, and label-mapping docs.

## Remaining Issues

- No trained CNN checkpoint is committed. TensorFlow is optional and absent in the current local environment, so CNN training was implemented but not executed here.
- Current committed model artifacts are random-forest baselines only.
- Grad-CAM will produce real heatmaps only after a CNN artifact is trained and available.
- Frontend metrics cards still use fallback display values until `reports/comparison.json` is generated and wired into a live metrics API view.
- Browser history is localStorage-only; production audit trails should be server-side.
- `frontend/src` still contains legacy Vite scaffold files. They are not active in the Next app but remain because the request said to keep current modules.
- Generated/vendor/data artifacts are present in the workspace (`frontend/node_modules`, `.next`, raw images, old reports). Ignore rules now cover them, but cleanup should be done deliberately in a real git repo.
- PDF layout was generated and smoke-tested, but Poppler rendering was not available for visual PDF page inspection in this environment.

## Suggested Future Improvements

- Train and compare EfficientNet-B0, EfficientNet-B3, and ResNet50 on the full dataset with TensorFlow installed.
- Add calibration curves, decision-curve analysis, and threshold tuning for deployment-specific cameras.
- Persist cases, reports, model versions, and audit events in a database.
- Add authentication, role-based access, PHI handling policy, and secure storage before any real clinical deployment.
- Add model cards and dataset cards for every trained artifact.
- Wire frontend Metrics and Model Comparison pages directly to `/metrics` with generated artifacts.
- Render representative PDFs to PNG in CI when Poppler is available.
- Remove legacy Vite scaffold files after confirming no one depends on them.