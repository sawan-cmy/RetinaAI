from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .constants import DR_CLASSES
from .datasets import DATASET_SPECS


def _json_block(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False)


def _metric_lines(metrics: Mapping[str, Any]) -> list[str]:
    keys = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "auc_ovr_macro", "false_negative_rate_any_dr", "train_seconds"]
    lines = ["| Metric | Value |", "|---|---:|"]
    for key in keys:
        if key in metrics:
            lines.append(f"| {key} | {metrics[key]} |")
    return lines


def write_model_card(
    model_name: str,
    model_path: str | Path,
    metrics: Mapping[str, Any],
    dataset_name: str,
    output_dir: str | Path = "reports/cards",
    checkpoint_path: str | Path | None = None,
    calibration_artifacts: Mapping[str, str] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{model_name}_model_card.md"
    calibration_artifacts = calibration_artifacts or {}
    text = f"""# RetinaAI Model Card: {model_name}

## Intended Use

Screening workflow support for diabetic retinopathy severity grading. This model is not a diagnostic medical device and requires qualified clinical review before care decisions.

## Model Artifact

- Model path: `{model_path}`
- Best checkpoint: `{checkpoint_path or model_path}`
- Training dataset: `{dataset_name}`
- Classes: `{_json_block(DR_CLASSES)}`

## Performance

{chr(10).join(_metric_lines(metrics))}

## Calibration And Thresholds

- Calibration JSON: `{calibration_artifacts.get('calibration_json', 'not generated')}`
- Calibration plot: `{calibration_artifacts.get('calibration_plot', 'not generated')}`
- Tuned thresholds: `{calibration_artifacts.get('thresholds_yaml', 'not generated')}`

## Known Limits

- Performance is dataset- and camera-dependent; use deployment-site calibration before clinical workflow use.
- External validation metrics must remain separate from internal test metrics.
- Grad-CAM explanations depend on a trained CNN checkpoint; baseline fallback cannot produce true CNN heatmaps.
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_dataset_card(
    dataset_name: str,
    labels_csv: str | Path | None,
    image_dir: str | Path | None,
    distributions: Mapping[str, Any],
    label_mapping: Mapping[int | str, int] | None,
    output_dir: str | Path = "reports/cards",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = DATASET_SPECS.get(dataset_name.lower())
    path = output_dir / f"{dataset_name}_dataset_card.md"
    mapping = label_mapping if label_mapping is not None else (spec.label_mapping if spec else {})
    notes = spec.mapping_notes if spec else "Custom dataset mapping supplied by caller."
    text = f"""# RetinaAI Dataset Card: {dataset_name}

## Source Contract

- Labels CSV: `{labels_csv or 'not recorded'}`
- Image directory: `{image_dir or 'not recorded'}`
- Loader key: `{dataset_name}`
- Mapping notes: {notes}

## Label Mapping

```json
{_json_block({str(key): int(value) for key, value in (mapping or {}).items()})}
```

## Split Distributions

```json
{_json_block(distributions)}
```

## Use Constraints

Raw dataset files stay outside git. Before publishing external validation metrics, confirm dataset license/access terms and document any source-schema conversion used for the labels.
"""
    path.write_text(text, encoding="utf-8")
    return path