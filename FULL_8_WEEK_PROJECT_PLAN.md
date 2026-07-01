# Full 8-Week Flagship Plan: AI Eye Disease Detection

## Project Target

Build a complete, research-style, clinical-screening prototype:

**Uncertainty-Aware Explainable AI Retinal Disease Screening System**

Primary task:

- Diabetic retinopathy severity grading from retinal fundus images.

Primary output:

- A deployed app, trained models, metrics dashboard, external validation, Grad-CAM explanations, uncertainty/manual-review routing, generated screening reports, clean GitHub repository, and IEEE-style research paper.

Medical positioning:

- This is an AI screening prototype, not a diagnostic medical device.
- Every UI, report, README, and paper must state that clinical diagnosis requires qualified medical review.

## Final System Scope

The finished system must support this full workflow:

```text
Upload retina image
-> image quality gate
-> preprocessing
-> DR severity prediction
-> uncertainty score
-> Grad-CAM explanation
-> manual-review routing
-> dashboard logging
-> AI screening report
```

DR classes:

- 0: No DR
- 1: Mild
- 2: Moderate
- 3: Severe
- 4: Proliferative DR

Minimum final feature set:

- Image quality rejection.
- Severity grading.
- Model comparison.
- Grad-CAM explainability.
- Confidence and entropy-based uncertainty.
- Manual-review routing.
- Internal test metrics.
- External validation.
- Streamlit dashboard.
- One-page AI screening report.
- IEEE-style research paper.
- Dockerfile and clean README.

## Repository Structure

```text
RetinaAI/
|-- app/
|   `-- streamlit_app.py
|-- configs/
|   |-- train.yaml
|   `-- thresholds.yaml
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- external/
|-- docs/
|   |-- architecture.md
|   |-- dataset_notes.md
|   `-- label_mapping.md
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_baseline_training.ipynb
|   `-- 03_error_analysis.ipynb
|-- reports/
|   |-- figures/
|   |-- paper/
|   `-- sample_reports/
|-- src/
|   |-- preprocessing.py
|   |-- quality_check.py
|   |-- datasets.py
|   |-- models.py
|   |-- train.py
|   |-- evaluate.py
|   |-- uncertainty.py
|   |-- gradcam.py
|   |-- inference.py
|   `-- report_generator.py
|-- tests/
|-- models/
|-- requirements.txt
|-- README.md
`-- Dockerfile
```

## Dataset Strategy

Main training dataset:

- APTOS 2019 Blindness Detection, preferred first dataset for 5-class DR severity grading.

Larger optional training dataset:

- EyePACS/Kaggle Diabetic Retinopathy Detection.

External validation:

- IDRiD, Messidor, or Messidor-2.

Image quality:

- EyeQ if available.
- If EyeQ access is delayed, implement heuristic quality gating first and document it as a limitation.

Dataset rules:

- Keep raw data unchanged.
- Store processed data separately.
- Document dataset source, license/access rules, class labels, label mapping, preprocessing, and imbalance.
- Do not mix DR severity, multi-disease, glaucoma, and quality labels without a written mapping table.

## Core Engineering Contracts

Single inference entry point:

```python
result = screen_retina_image(image_path)
```

Expected result shape:

```json
{
  "quality": {
    "status": "accepted",
    "blur_score": 142.0,
    "brightness_score": 118.5,
    "contrast_score": 51.2
  },
  "prediction": {
    "class_id": 2,
    "class_name": "Moderate DR",
    "confidence": 0.87,
    "probabilities": [0.02, 0.04, 0.87, 0.05, 0.02]
  },
  "uncertainty": {
    "entropy": 0.41,
    "margin": 0.82,
    "manual_review": false,
    "reason": "confidence_above_threshold"
  },
  "outputs": {
    "gradcam_path": "reports/figures/example_gradcam.png",
    "report_path": "reports/sample_reports/sample_ai_report.pdf"
  }
}
```

Manual-review triggers:

- Quality gate rejects image.
- Confidence below threshold.
- Entropy above threshold.
- Top-2 probability margin below threshold.
- Model ensemble disagreement, if ensemble is implemented.

## Week 1 - Data and Baseline

Goal:

- Establish the data pipeline and first real baseline model.

Build:

- Dataset download instructions.
- Raw/processed data layout.
- Label loading and class distribution report.
- Preprocessing: black border crop, resize, normalize.
- Train/validation/test split.
- Baseline MobileNetV2 or ResNet50 training.
- First confusion matrix.

Deliverables:

- `docs/dataset_notes.md`
- `docs/label_mapping.md`
- `src/preprocessing.py`
- `src/datasets.py`
- `src/train.py`
- Baseline model checkpoint.
- First metrics JSON/CSV.

Acceptance criteria:

- A baseline model trains end to end.
- Class imbalance is measured.
- Split strategy is documented.
- No paper claims are made beyond real measured results.

## Week 2 - Strong Model Comparison

Goal:

- Turn the classifier into a proper experiment.

Build:

- Training config.
- Shared model factory.
- Reproducible training runs.
- Compare MobileNetV2, ResNet50, EfficientNetB0/B3.
- Optional advanced model: ConvNeXt or ViT only if compute allows.
- Save metrics and latency for each model.

Metrics:

- Accuracy.
- Macro F1.
- Weighted F1.
- Precision.
- Recall.
- AUC if implemented correctly.
- False-negative rate.
- Latency per image.

Deliverables:

- `configs/train.yaml`
- `src/models.py`
- `src/evaluate.py`
- `reports/figures/model_comparison.png`
- `reports/figures/confusion_matrix_best_model.png`
- Model comparison table.

Acceptance criteria:

- Best model is chosen by macro F1 and false-negative behavior, not accuracy alone.
- Every model result is reproducible from a command.

## Week 3 - Image Quality and Uncertainty

Goal:

- Add safety-aware screening behavior.

Build:

- Blur detection.
- Brightness detection.
- Contrast detection.
- Retina visibility heuristic.
- Confidence threshold.
- Entropy score.
- Top-2 margin score.
- Manual-review routing.

Deliverables:

- `src/quality_check.py`
- `src/uncertainty.py`
- `configs/thresholds.yaml`
- Quality gate examples in `reports/figures/`

Acceptance criteria:

- Bad images are rejected before prediction.
- Low-confidence predictions are routed to manual review.
- Thresholds are configurable.
- README explains that quality/uncertainty routing reduces unsafe blind prediction.

## Week 4 - Explainability

Goal:

- Make the model decision visually inspectable.

Build:

- Grad-CAM for best CNN model.
- Heatmap overlay.
- Batch generation of Grad-CAM samples.
- Correct and failed case examples.

Deliverables:

- `src/gradcam.py`
- Grad-CAM examples for each severity class.
- Error-analysis examples.

Acceptance criteria:

- The app can show original image and Grad-CAM overlay.
- Paper has a figure showing representative Grad-CAM outputs.
- Grad-CAM is described as decision support, not proof of diagnosis.

## Week 5 - Dashboard and Report Generator

Goal:

- Turn the research pipeline into a product-style screening app.

Build:

- Streamlit app.
- Upload workflow.
- Screening result panel.
- Quality metrics panel.
- Confidence/uncertainty panel.
- Grad-CAM display.
- Metrics dashboard.
- One-page report generator.

Deliverables:

- `app/streamlit_app.py`
- `src/inference.py`
- `src/report_generator.py`
- `reports/sample_reports/sample_ai_report.pdf`
- App screenshots.

Acceptance criteria:

- App runs locally.
- User can upload a retinal image and get a full screening result.
- Report includes disclaimer, quality status, predicted severity, confidence, uncertainty, manual-review recommendation, and Grad-CAM image.

## Week 6 - External Validation

Goal:

- Show generalization beyond one dataset.

Build:

- External dataset loader.
- Label mapping for IDRiD/Messidor/Messidor-2.
- External evaluation command.
- Generalization gap table.
- Failure case analysis.

Deliverables:

- `data/external/README.md`
- External validation metrics.
- `reports/figures/external_validation_matrix.png`
- Error-analysis notes.

Acceptance criteria:

- External validation is reported separately from internal test results.
- Label mapping is documented.
- The paper discusses domain shift and limitations honestly.

## Week 7 - Paper, README, and Research Packaging

Goal:

- Make the project defensible in interviews and review.

Paper title:

**Uncertainty-Aware Explainable Deep Learning Framework for Diabetic Retinopathy Severity Screening from Retinal Fundus Images**

Paper sections:

- Abstract.
- Introduction.
- Related Work.
- Dataset and Preprocessing.
- Methodology.
- Results.
- Discussion.
- Limitations.
- Conclusion.

Build:

- IEEE-style paper draft.
- Architecture diagram.
- Model comparison table.
- Confusion matrix figure.
- ROC/AUC figure if valid.
- Grad-CAM figure.
- External validation table.
- README with setup, commands, screenshots, and limitations.

Deliverables:

- `reports/paper/research_paper.pdf`
- `README.md`
- `docs/architecture.md`
- Final figures.

Acceptance criteria:

- Paper numbers match generated metrics.
- README can be followed by another developer.
- Limitations are explicit.

## Week 8 - Deployment and Final Polish

Goal:

- Ship the final portfolio-grade version.

Build:

- Dockerfile.
- Deployment setup.
- Demo video script.
- Final screenshots.
- Basic tests/self-checks.
- Repo cleanup.
- Reproducibility commands.

Deliverables:

- `Dockerfile`
- Live demo link if deployment target is available.
- Demo video or script.
- Final GitHub-ready repository.

Acceptance criteria:

- App starts from documented commands.
- Inference works on sample images.
- Tests/self-checks pass.
- Paper, report, figures, models, and README are linked cleanly.

## Final Checks Before Resume

- GitHub repo has clean README, architecture diagram, screenshots, and setup instructions.
- Live demo works or local demo instructions are reliable.
- Paper PDF is included under `reports/paper/`.
- Confusion matrix, model comparison table, Grad-CAM examples, and sample AI report are visible.
- Dataset licenses/access rules are documented.
- You can explain every metric and limitation.

## Final Resume Line

Built an uncertainty-aware explainable retinal screening platform for diabetic retinopathy severity grading using transfer-learning vision models, image quality assessment, Grad-CAM explainability, uncertainty-based manual-review routing, external validation, dashboard deployment, automated AI screening reports, and an IEEE-style research paper.

## Non-Negotiables

- No fake clinical claims.
- No paper metrics without real experiment outputs.
- No blind prediction on poor-quality images.
- No accuracy-only evaluation.
- No dataset mixing without label mapping.
- No new dependency unless standard library or existing dependencies cannot solve the task.
