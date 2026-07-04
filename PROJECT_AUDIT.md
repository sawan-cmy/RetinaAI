# Project Audit

Date: 2026-07-02

## Audit Scope

Reviewed repository source, configuration, docs, tests, frontend app code, package metadata, Docker setup, CI, generated artifact paths, and current git state. Vendor/build/raw-data/generated directories were identified but not hand-audited as source: `frontend/node_modules`, `frontend/.next`, `.keras-cache`, `data/raw`, Python `__pycache__`, generated reports, generated model artifacts, and generated test outputs.

## Findings Before Fixes

- Random forest existed as a baseline, but the production pipeline needed CNN training, model comparison, external validation, calibration, and artifact packaging connected end to end.
- Dataset loading had to support APTOS, EyePACS, Messidor, and IDRiD-style label schemas instead of assuming APTOS only.
- Grad-CAM needed to be generated from CNN checkpoints and fall back explicitly when unavailable.
- Inference needed to route through preprocessing, quality gate, model selection, fallback, uncertainty, recommendation, Grad-CAM, and PDF report without crashing on missing/corrupt models.
- FastAPI, case history, role-aware API keys, frontend proxying, Docker, CI, pre-commit, and frontend e2e checks needed production wiring.
- The frontend had professional pages, but metrics/model-comparison views still used static fallback values even when backend metrics artifacts existed.
- The frontend artifact API compared a resolved file path against a non-resolved reports root on Windows, causing valid artifact reads to be rejected.
- `PROJECT_AUDIT.md` had stale text claiming git was unavailable; this workspace is a git repo with a dirty worktree and generated artifacts.

## Fixed

- Added dataset abstraction in `src.datasets` for APTOS, APTOS 2019, EyePACS, Messidor, and IDRiD-style CSVs with reusable label mapping files.
- Implemented TensorFlow/Keras transfer-learning support for EfficientNet-B0, EfficientNet-B3, and ResNet50 while preserving the random forest as a baseline/fallback.
- Added checkpoint saving, metadata saving/loading, reproducible seeds, class weighting, early stopping, mixed precision, and learning-rate reduction in `src.train`.
- Implemented Keras Grad-CAM heatmap generation and overlay saving for CNN artifacts, with explicit unavailable explanation images for fallback/missing models.
- Upgraded inference to run quality gate, CNN or fallback prediction, confidence, predictive entropy, top-2 margin, recommendation, latency capture, Grad-CAM/explanation artifact, and PDF report generation.
- Added calibration and site-threshold tuning artifacts, model cards, dataset cards, and training artifact packaging.
- Added model comparison output: `reports/comparison.csv`, `reports/comparison.json`, `reports/comparison.png`, model-comparison aliases, metrics, latency, training time, AUC, F1, and false-negative rate.
- Added external validation command for supported datasets with label mapping and per-site calibration output.
- Added FastAPI endpoints for `/health`, `/predict`, `/quality`, `/gradcam`, `/report`, `/metrics`, `/models`, `/cases`, `/cases/{run_id}`, and report artifacts.
- Added server-side case history through SQLite and API-key role handling for viewer/clinician/admin workflows.
- Upgraded PDF reports with patient ID, date, quality metrics, prediction, confidence, uncertainty, Grad-CAM/explanation, recommendation, disclaimer, and hospital-style layout.
- Reworked the frontend into a Next.js healthcare dashboard with Upload, Prediction, Grad-CAM, Metrics, Reports, Settings, Model Comparison, and History pages.
- Replaced Python shelling from the frontend upload path with a FastAPI proxy route.
- Added a frontend `/api/metrics` proxy and wired Metrics and Model Comparison pages to generated backend metrics with explicit fallback display when no metrics artifact exists.
- Fixed the frontend artifact API reports-root resolution bug and added a Playwright regression test for serving allowed report files and blocking traversal.
- Added pytest coverage for preprocessing, quality, training baseline, inference fallback, Grad-CAM overlay behavior, uncertainty routing, calibration, case storage, and FastAPI endpoints.
- Added Playwright e2e coverage for dashboard navigation and artifact API behavior.
- Added Docker, Docker Compose, GitHub Actions, lint/typecheck/build/e2e checks, pre-commit hooks, and updated README/docs.

## Verification

- `python -m pytest` passed: 21 tests.
- `python tests/self_check.py` passed.
- `python -m compileall -q app src tests` exited successfully; local generated temp directories under `tests` still print access-denied listing warnings.
- `npm run lint` passed.
- `npm run typecheck` passed.
- `npm run build` passed outside the sandbox after the sandbox produced `spawn EPERM`.
- `npm run test:e2e` passed outside the sandbox: 2 Playwright tests.

## Remaining Issues

- No trained CNN checkpoint is committed. TensorFlow is optional in the base setup, so CNN training support is implemented but full EfficientNet/ResNet training was not executed here.
- Grad-CAM produces real heatmaps only after a trained CNN artifact is available; baseline fallback intentionally produces an unavailable-explanation image.
- Current committed/generated model artifacts are local baseline/smoke artifacts and should not be treated as publishable clinical performance evidence.
- Frontend Metrics and Model Comparison pages now consume `/metrics`, but they still show static fallback values when the FastAPI backend is unreachable or no comparison artifact exists.
- `frontend/src` still contains legacy Vite scaffold files. They are inactive in the Next app and kept because the request said to keep current modules.
- Generated/cache artifacts are present in the workspace, including `.keras-cache`, `models`, `reports`, `frontend/.next`, and access-restricted old pytest temp directories. Cleanup should be deliberate and separate from this upgrade.
- PDF layout is smoke-tested through report generation, but Poppler-based visual PDF rendering is not configured in this environment.
- Prototype API-key auth and SQLite storage are not sufficient for real PHI production deployment without identity, encryption, retention, and audit controls.

## Suggested Future Improvements

- Train EfficientNet-B0, EfficientNet-B3, and ResNet50 on the full target dataset with TensorFlow/GPU support, then regenerate comparison, calibration, model-card, dataset-card, and package artifacts.
- Run external validation for EyePACS, Messidor, and IDRiD with verified source label schemas and site/camera-specific thresholds.
- Add CI PDF render checks when Poppler is available.
- Replace prototype API-key auth with a production identity provider and encrypted case/report storage before handling real clinical data.
- Remove legacy Vite scaffold files and generated temp/cache directories in a dedicated cleanup change once stakeholders confirm they are not needed.
- Add live model promotion governance around signed artifacts, model version pinning, rollback, and audit events.