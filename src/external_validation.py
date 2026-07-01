from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .datasets import load_labels
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


def validate_external_dataset(
    model_path: str | Path,
    labels_csv: str | Path,
    image_dir: str | Path,
    dataset: str,
    output_path: str | Path | None = None,
    label_mapping: dict | None = None,
    max_images: int | None = None,
) -> dict:
    df = load_labels(labels_csv, image_dir, dataset=dataset, label_mapping=label_mapping)
    if max_images:
        df = df.sample(n=min(max_images, len(df)), random_state=42).reset_index(drop=True)

    bundle = load_model(model_path)
    started = perf_counter()
    if bundle.get("kind") == "keras_cnn":
        probabilities = np.vstack([predict_image_probabilities(bundle, path) for path in df["path"]])
    else:
        probabilities = np.vstack([predict_probabilities(bundle, extract_handcrafted_features(path)) for path in df["path"]])
    latency_ms = ((perf_counter() - started) / max(1, len(df))) * 1000
    metrics = calculate_metrics(df["label"].to_numpy(dtype=int), probabilities)
    metrics.update(
        {
            "dataset": dataset,
            "sample_count": int(len(df)),
            "latency_ms": round(float(latency_ms), 3),
            "model_path": str(model_path),
            "label_mapping": label_mapping or {},
        }
    )
    output = Path(output_path or f"reports/external_validation_{dataset}.json")
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
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())