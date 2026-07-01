from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .constants import DR_CLASSES


def false_negative_rate(y_true, y_pred) -> float | None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    positive = y_true > 0
    if not positive.any():
        return None
    return float(((y_pred == 0) & positive).sum() / positive.sum())


def calculate_metrics(y_true, probabilities, labels: list[int] | None = None) -> dict:
    labels = labels or list(DR_CLASSES)
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = probabilities.argmax(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "false_negative_rate_any_dr": false_negative_rate(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }

    try:
        y_bin = label_binarize(y_true, classes=labels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = float(roc_auc_score(y_bin, probabilities[:, labels], average="macro", multi_class="ovr"))
        metrics["auc_ovr_macro"] = auc if np.isfinite(auc) else None
    except Exception:
        metrics["auc_ovr_macro"] = None
    return metrics


def save_metrics(metrics: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return path


def plot_confusion_matrix(matrix, path: str | Path, labels: list[int] | None = None) -> Path:
    labels = labels or list(DR_CLASSES)
    matrix = np.asarray(matrix)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)), [DR_CLASSES[i] for i in labels], rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), [DR_CLASSES[i] for i in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path




