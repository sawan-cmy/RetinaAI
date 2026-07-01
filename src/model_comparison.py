from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from .datasets import class_distribution, load_labels, split_dataframe
from .evaluate import calculate_metrics, plot_confusion_matrix
from .models import CNN_MODEL_SPECS, make_baseline_model, predict_image_probabilities, predict_probabilities, save_model
from .preprocessing import extract_handcrafted_features
from .train import train_cnn_from_dataframe

BASELINE_CONFIG = {"name": "baseline_sklearn", "n_estimators": 300, "max_depth": None, "seed": 42}
DEFAULT_MODEL_NAMES = ["baseline_sklearn", "efficientnet_b0", "efficientnet_b3", "resnet50"]


def _features(paths) -> np.ndarray:
    return np.vstack([extract_handcrafted_features(path) for path in paths])


def _latency_ms(bundle: dict, rows, limit: int = 32, image_mode: bool = False) -> float:
    sample = list(rows[: max(1, min(limit, len(rows)))])
    started = perf_counter()
    for row in sample:
        if image_mode:
            predict_image_probabilities(bundle, row)
        else:
            predict_probabilities(bundle, row)
    return round(((perf_counter() - started) / len(sample)) * 1000, 3)


def _flat_row(name: str, metrics: dict, latency_ms: float | None, model_path: str | None, status: str = "trained", error: str | None = None) -> dict:
    return {
        "name": name,
        "status": status,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("macro_precision"),
        "recall": metrics.get("macro_recall"),
        "f1": metrics.get("macro_f1"),
        "auc": metrics.get("auc_ovr_macro"),
        "latency_ms": latency_ms,
        "training_time_seconds": metrics.get("train_seconds"),
        "false_negative_rate": metrics.get("false_negative_rate_any_dr"),
        "model_path": model_path,
        "confusion_matrix": metrics.get("confusion_matrix"),
        "error": error,
    }


def _write_csv(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "status",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "latency_ms",
        "training_time_seconds",
        "false_negative_rate",
        "model_path",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return path


def _plot_comparison(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    trained = [row for row in rows if row.get("status") == "trained"] or rows
    names = [row["name"] for row in trained]
    f1_scores = [row.get("f1") or 0 for row in trained]
    false_negative = [row.get("false_negative_rate") or 0 for row in trained]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(trained))
    width = 0.35
    ax.bar(x - width / 2, f1_scores, width, label="Macro F1")
    ax.bar(x + width / 2, false_negative, width, label="Any-DR false negative rate")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("RetinaAI model comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _train_baseline(train_df, test_df, out_dir: Path) -> dict:
    x_train = _features(train_df["path"])
    y_train = train_df["label"].to_numpy(dtype=int)
    x_test = _features(test_df["path"])
    y_test = test_df["label"].to_numpy(dtype=int)
    model = make_baseline_model(
        n_estimators=BASELINE_CONFIG["n_estimators"],
        max_depth=BASELINE_CONFIG["max_depth"],
        seed=BASELINE_CONFIG["seed"],
    )
    started = perf_counter()
    model.fit(x_train, y_train)
    train_seconds = round(perf_counter() - started, 3)
    bundle = {"kind": "sklearn_random_forest", "name": BASELINE_CONFIG["name"], "model": model, "classes": [0, 1, 2, 3, 4]}
    probabilities = np.vstack([predict_probabilities(bundle, row) for row in x_test])
    metrics = calculate_metrics(y_test, probabilities)
    metrics["train_seconds"] = train_seconds
    model_path = save_model(bundle, out_dir / "baseline_sklearn.pkl")
    return _flat_row(BASELINE_CONFIG["name"], metrics, _latency_ms(bundle, x_test), str(model_path))


def compare_models(
    labels_csv: str | Path,
    image_dir: str | Path,
    max_images: int | None = None,
    out_dir: str | Path = "models",
    dataset: str = "aptos",
    model_names: list[str] | None = None,
    epochs: int = 8,
    batch_size: int = 32,
) -> dict:
    model_names = model_names or DEFAULT_MODEL_NAMES
    df = load_labels(labels_csv, image_dir, dataset=dataset)
    if max_images:
        df = df.sample(n=min(max_images, len(df)), random_state=42).reset_index(drop=True)

    train_df, val_df, test_df = split_dataframe(df)
    rows: list[dict] = []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in model_names:
        normalized = name.strip().lower()
        if normalized in {"baseline", "baseline_sklearn", "random_forest", "rf"}:
            rows.append(_train_baseline(train_df, test_df, out_dir))
            continue
        if normalized not in CNN_MODEL_SPECS:
            rows.append(_flat_row(normalized, {}, None, None, status="skipped", error="unsupported_model"))
            continue
        try:
            metrics = train_cnn_from_dataframe(
                train_df,
                val_df,
                test_df,
                model_name=normalized,
                out_dir=out_dir,
                epochs=epochs,
                batch_size=batch_size,
            )
            rows.append(_flat_row(normalized, metrics, None, metrics.get("model_path")))
        except Exception as exc:
            rows.append(_flat_row(normalized, {}, None, None, status="skipped", error=str(exc)))

    trained = [row for row in rows if row.get("status") == "trained"]
    best = sorted(trained, key=lambda row: (row.get("f1") or -1, -1 * (row.get("false_negative_rate") or 1)) , reverse=True)[0] if trained else None
    report = {
        "selection_rule": "highest macro F1, then lower any-DR false-negative rate",
        "best_model": best["name"] if best else None,
        "train_distribution": class_distribution(train_df),
        "validation_distribution": class_distribution(val_df),
        "test_distribution": class_distribution(test_df),
        "models": rows,
    }

    reports = Path("reports")
    figures = reports / "figures"
    reports.mkdir(exist_ok=True)
    for json_name in ("comparison.json", "model_comparison.json"):
        (reports / json_name).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    for csv_name in ("comparison.csv", "model_comparison.csv"):
        _write_csv(rows, reports / csv_name)
    _plot_comparison(rows, reports / "comparison.png")
    for figure_name in ("comparison.png", "model_comparison.png"):
        _plot_comparison(rows, figures / figure_name)
    if best and best.get("confusion_matrix"):
        plot_confusion_matrix(best["confusion_matrix"], figures / "confusion_matrix_best_model.png")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare RetinaAI baseline and transfer-learning model candidates.")
    parser.add_argument("--labels-csv", default="data/raw/aptos2019/train.csv")
    parser.add_argument("--image-dir", default="data/raw/aptos2019/images_288_scaled")
    parser.add_argument("--dataset", default="aptos")
    parser.add_argument("--models", default=",".join(DEFAULT_MODEL_NAMES))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()
    names = [name.strip() for name in args.models.split(",") if name.strip()]
    print(json.dumps(compare_models(args.labels_csv, args.image_dir, args.max_images, args.out_dir, args.dataset, names, args.epochs, args.batch_size), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())