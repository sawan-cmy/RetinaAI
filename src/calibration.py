from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .uncertainty import DEFAULT_THRESHOLDS, predictive_entropy, top2_margin


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "_", value.strip().lower()).strip("_") or "default"


def probability_matrix(probabilities) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a 2D array with at least two classes")
    if not np.isfinite(matrix).all():
        raise ValueError("probabilities must be finite")
    totals = matrix.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("each probability row must sum to a positive value")
    return matrix / totals[:, None]


def confidence_calibration(y_true, probabilities, n_bins: int = 10) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probs = probability_matrix(probabilities)
    if len(y_true) != len(probs):
        raise ValueError("y_true and probabilities must have the same length")

    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, Any]] = []
    ece = 0.0
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= low) & (confidence <= high if index == n_bins - 1 else confidence < high)
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            ece += (count / len(confidence)) * abs(accuracy - mean_confidence)
        else:
            mean_confidence = None
            accuracy = None
        bins.append(
            {
                "bin_low": round(float(low), 6),
                "bin_high": round(float(high), 6),
                "count": count,
                "mean_confidence": None if mean_confidence is None else round(mean_confidence, 6),
                "accuracy": None if accuracy is None else round(accuracy, 6),
            }
        )
    return {"n": int(len(y_true)), "n_bins": int(n_bins), "ece": round(float(ece), 6), "bins": bins}


def plot_calibration_curve(curve: dict[str, Any], path: str | Path, title: str = "Calibration curve") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [row["mean_confidence"] for row in curve["bins"] if row["mean_confidence"] is not None]
    ys = [row["accuracy"] for row in curve["bins"] if row["accuracy"] is not None]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="#64748B", label="Perfect calibration")
    if xs:
        ax.plot(xs, ys, marker="o", color="#1D4ED8", label="Observed")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title} (ECE {curve['ece']:.3f})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _quantile_candidates(values: np.ndarray, quantiles: list[float], include: list[float]) -> list[float]:
    candidates = [float(value) for value in include]
    if len(values):
        candidates.extend(float(value) for value in np.quantile(values, quantiles))
    return sorted({round(value, 6) for value in candidates if np.isfinite(value)})


def _operating_point(y_true: np.ndarray, probabilities: np.ndarray, thresholds: dict[str, float]) -> dict[str, Any]:
    confidence = probabilities.max(axis=1)
    entropy = np.asarray([predictive_entropy(row) for row in probabilities], dtype=float)
    margin = np.asarray([top2_margin(row) for row in probabilities], dtype=float)
    y_pred = probabilities.argmax(axis=1)
    manual_review = (
        (confidence < float(thresholds["min_confidence"]))
        | (entropy > float(thresholds["max_entropy"]))
        | (margin < float(thresholds["min_top2_margin"]))
    )
    positive = y_true > 0
    released = ~manual_review
    auto_false_negatives = (released & positive & (y_pred == 0)).sum()
    positive_count = int(positive.sum())
    released_count = int(released.sum())
    return {
        "review_rate": round(float(manual_review.mean()), 6),
        "auto_release_rate": round(float(released.mean()), 6),
        "auto_false_negative_rate_any_dr": None if positive_count == 0 else round(float(auto_false_negatives / positive_count), 6),
        "auto_accuracy": None if released_count == 0 else round(float((y_pred[released] == y_true[released]).mean()), 6),
        "auto_released_count": released_count,
        "manual_review_count": int(manual_review.sum()),
        "positive_count": positive_count,
    }


def tune_uncertainty_thresholds(
    y_true,
    probabilities,
    site: str = "default",
    max_false_negative_rate: float = 0.05,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probs = probability_matrix(probabilities)
    if len(y_true) != len(probs):
        raise ValueError("y_true and probabilities must have the same length")

    confidence = probs.max(axis=1)
    entropy = np.asarray([predictive_entropy(row) for row in probs], dtype=float)
    margin = np.asarray([top2_margin(row) for row in probs], dtype=float)
    # ponytail: quantile grid is O(k^3); switch to Bayesian/coordinate search only when site validation sets get large enough to matter.
    confidence_grid = _quantile_candidates(confidence, [0.05, 0.10, 0.25, 0.50, 0.75], [0.0, DEFAULT_THRESHOLDS["min_confidence"]])
    entropy_grid = _quantile_candidates(entropy, [0.50, 0.75, 0.90, 0.95, 1.0], [DEFAULT_THRESHOLDS["max_entropy"], float(entropy.max()) + 1e-6])
    margin_grid = _quantile_candidates(margin, [0.05, 0.10, 0.25, 0.50, 0.75], [0.0, DEFAULT_THRESHOLDS["min_top2_margin"]])

    candidates: list[dict[str, Any]] = []
    for min_confidence in confidence_grid:
        for max_entropy in entropy_grid:
            for min_top2_margin in margin_grid:
                thresholds = {
                    "min_confidence": float(min_confidence),
                    "max_entropy": float(max_entropy),
                    "min_top2_margin": float(min_top2_margin),
                }
                point = _operating_point(y_true, probs, thresholds)
                candidates.append({"thresholds": thresholds, **point})

    def fnr(row: dict[str, Any]) -> float:
        value = row["auto_false_negative_rate_any_dr"]
        return 0.0 if value is None else float(value)

    feasible = [row for row in candidates if fnr(row) <= max_false_negative_rate]
    if feasible:
        best = sorted(feasible, key=lambda row: (float(row["review_rate"]), fnr(row), -float(row["auto_release_rate"])))[0]
        status = "target_met"
    else:
        best = sorted(candidates, key=lambda row: (fnr(row), float(row["review_rate"])))[0]
        status = "target_not_met"

    return {
        "site": site,
        "status": status,
        "target_max_false_negative_rate_any_dr": float(max_false_negative_rate),
        "thresholds": {key: round(float(value), 6) for key, value in best["thresholds"].items()},
        "operating_point": {key: value for key, value in best.items() if key != "thresholds"},
        "candidate_count": len(candidates),
    }


def write_calibration_artifacts(
    y_true,
    probabilities,
    site: str,
    model_name: str,
    out_dir: str | Path = "reports/calibration",
    max_false_negative_rate: float = 0.05,
) -> dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(site)}_{_slug(model_name)}"
    curve = confidence_calibration(y_true, probabilities)
    tuned = tune_uncertainty_thresholds(y_true, probabilities, site=site, max_false_negative_rate=max_false_negative_rate)
    payload = {"site": site, "model_name": model_name, "calibration": curve, "threshold_tuning": tuned}

    json_path = out_dir / f"{stem}_calibration.json"
    png_path = out_dir / f"{stem}_calibration.png"
    thresholds_path = out_dir / f"{stem}_thresholds.yaml"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    plot_calibration_curve(curve, png_path, title=f"{site} / {model_name}")
    thresholds_path.write_text(yaml.safe_dump({"sites": {site: {"uncertainty": tuned["thresholds"]}}}, sort_keys=False), encoding="utf-8")
    return {"calibration_json": str(json_path), "calibration_plot": str(png_path), "thresholds_yaml": str(thresholds_path)}


def _load_predictions(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        y_true = [row["label"] for row in rows]
        probabilities = [row.get("probabilities") or [row[f"p{index}"] for index in range(5)] for row in rows]
        return np.asarray(y_true, dtype=int), probability_matrix(probabilities)

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"no prediction rows in {path}")
    probability_fields = [field for field in rows[0] if field and field.lower().startswith("p")]
    probability_fields = sorted(probability_fields, key=lambda value: int(value[1:]) if value[1:].isdigit() else value)
    y_true = [row["label"] for row in rows]
    probabilities = [[row[field] for field in probability_fields] for row in rows]
    return np.asarray(y_true, dtype=int), probability_matrix(probabilities)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate calibration curves and tune uncertainty thresholds for a deployment site.")
    parser.add_argument("--predictions", required=True, help="CSV/JSON with label and p0..p4 or probabilities fields.")
    parser.add_argument("--site", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", default="reports/calibration")
    parser.add_argument("--max-fnr", type=float, default=0.05)
    args = parser.parse_args()
    y_true, probabilities = _load_predictions(args.predictions)
    artifacts = write_calibration_artifacts(y_true, probabilities, args.site, args.model, args.out_dir, args.max_fnr)
    print(json.dumps(artifacts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())