from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .calibration import write_calibration_artifacts
from .datasets import DATASET_SPECS, load_labels
from .evaluate import calculate_metrics, save_metrics
from .models import load_model, predict_image_probabilities, predict_probabilities
from .preprocessing import extract_handcrafted_features


def _load_label_mapping(value: str | None) -> dict | None:
    if not value:
        return None
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def _mapping_for(dataset: str, label_mapping: dict | None) -> dict:
    if label_mapping is not None:
        return {str(key): int(value) for key, value in label_mapping.items()}
    spec = DATASET_SPECS.get(dataset.lower())
    return {str(key): int(value) for key, value in (spec.label_mapping if spec else {}).items()}


def _save_predictions(df, probabilities: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["dataset", "image_id", "label", "prediction", *[f"p{index}" for index in range(probabilities.shape[1])]]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, probs in zip(df.itertuples(index=False), probabilities):
            writer.writerow(
                {
                    "dataset": row.dataset,
                    "image_id": row.image_id,
                    "label": int(row.label),
                    "prediction": int(np.argmax(probs)),
                    **{f"p{index}": round(float(value), 8) for index, value in enumerate(probs)},
                }
            )
    return output_path


def validate_external_dataset(
    model_path: str | Path,
    labels_csv: str | Path,
    image_dir: str | Path,
    dataset: str,
    output_path: str | Path | None = None,
    label_mapping: dict | None = None,
    max_images: int | None = None,
    site: str | None = None,
    max_false_negative_rate: float = 0.05,
) -> dict:
    mapping = _mapping_for(dataset, label_mapping)
    df = load_labels(labels_csv, image_dir, dataset=dataset, label_mapping=mapping)
    if max_images:
        df = df.sample(n=min(max_images, len(df)), random_state=42).reset_index(drop=True)

    bundle = load_model(model_path)
    started = perf_counter()
    if bundle.get("kind") in {"keras_cnn", "torch_cnn"}:
        probabilities = np.vstack([predict_image_probabilities(bundle, path) for path in df["path"]])
    else:
        probabilities = np.vstack([predict_probabilities(bundle, extract_handcrafted_features(path)) for path in df["path"]])
    latency_ms = ((perf_counter() - started) / max(1, len(df))) * 1000
    metrics = calculate_metrics(df["label"].to_numpy(dtype=int), probabilities)
    site = site or dataset
    out_dir = Path("reports/external_validation")
    predictions_path = _save_predictions(df, probabilities, out_dir / f"{dataset}_predictions.csv")
    calibration = write_calibration_artifacts(
        df["label"].to_numpy(dtype=int),
        probabilities,
        site=site,
        model_name=Path(model_path).stem,
        out_dir=out_dir,
        max_false_negative_rate=max_false_negative_rate,
    )
    metrics.update(
        {
            "dataset": dataset,
            "site": site,
            "sample_count": int(len(df)),
            "latency_ms": round(float(latency_ms), 3),
            "model_path": str(model_path),
            "label_mapping": mapping,
            "predictions_path": str(predictions_path),
            "calibration_artifacts": calibration,
        }
    )
    output = Path(output_path or out_dir / f"{dataset}_metrics.json")
    save_metrics(metrics, output)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run external validation for APTOS, EyePACS, Messidor, or IDRiD-style label CSVs.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=["aptos", "aptos2019", "eyepacs", "messidor", "idrid"])
    parser.add_argument("--label-map", default=None, help="JSON object or path mapping source labels to 0-4 DR labels")
    parser.add_argument("--output", default=None)
    parser.add_argument("--site", default=None, help="Deployment site/camera identifier used for calibration artifacts")
    parser.add_argument("--max-fnr", type=float, default=0.05, help="Maximum auto-released any-DR false-negative rate target")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()
    result = validate_external_dataset(
        args.model,
        args.labels_csv,
        args.image_dir,
        args.dataset,
        args.output,
        _load_label_mapping(args.label_map),
        args.max_images,
        args.site,
        args.max_fnr,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())