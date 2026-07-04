# RetinaAI Model Card: efficientnet_b0

## Intended Use

Screening workflow support for diabetic retinopathy severity grading. This model is not a diagnostic medical device and requires qualified clinical review before care decisions.

## Model Artifact

- Model path: `models\smoke\efficientnet_b0.keras`
- Best checkpoint: `models\smoke\efficientnet_b0_best.keras`
- Training dataset: `aptos_smoke`
- Classes: `{
  "0": "No DR",
  "1": "Mild DR",
  "2": "Moderate DR",
  "3": "Severe DR",
  "4": "Proliferative DR"
}`

## Performance

| Metric | Value |
|---|---:|
| accuracy | 0.4 |
| macro_f1 | 0.21333333333333332 |
| weighted_f1 | 0.3466666666666667 |
| macro_precision | 0.25 |
| macro_recall | 0.3 |
| auc_ovr_macro | None |
| false_negative_rate_any_dr | 0.0 |
| train_seconds | 20.851 |

## Calibration And Thresholds

- Calibration JSON: `reports\calibration\aptos_smoke_efficientnet_b0_calibration.json`
- Calibration plot: `reports\calibration\aptos_smoke_efficientnet_b0_calibration.png`
- Tuned thresholds: `reports\calibration\aptos_smoke_efficientnet_b0_thresholds.yaml`

## Known Limits

- Performance is dataset- and camera-dependent; use deployment-site calibration before clinical workflow use.
- External validation metrics must remain separate from internal test metrics.
- Grad-CAM explanations depend on a trained CNN checkpoint; baseline fallback cannot produce true CNN heatmaps.
