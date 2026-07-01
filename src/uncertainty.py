from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

DEFAULT_THRESHOLDS = {
    "min_confidence": 0.70,
    "max_entropy": 1.20,
    "min_top2_margin": 0.15,
}


def load_uncertainty_thresholds(path: str | Path | None = None) -> dict:
    if not path or not Path(path).exists():
        return DEFAULT_THRESHOLDS.copy()
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return {**DEFAULT_THRESHOLDS, **(data.get("uncertainty", data) or {})}


def normalize_probabilities(probabilities) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probs.size == 0 or not np.isfinite(probs).all():
        raise ValueError("probabilities must be a non-empty finite vector")
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("probabilities must sum to a positive value")
    return probs / total


def predictive_entropy(probabilities) -> float:
    probs = normalize_probabilities(probabilities)
    return float(-(probs * np.log(probs + 1e-12)).sum())


def top2_margin(probabilities) -> float:
    probs = np.sort(normalize_probabilities(probabilities))
    return float(probs[-1] - probs[-2]) if probs.size > 1 else 1.0


def route_case(quality_status: str, probabilities=None, thresholds: dict | None = None, unavailable_reason: str = "model_missing") -> dict:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    reasons: list[str] = []

    if quality_status != "accepted":
        reasons.append("quality_rejected")

    if probabilities is None:
        if unavailable_reason and unavailable_reason not in reasons:
            reasons.append(unavailable_reason)
        return {
            "manual_review": True,
            "reason": ",".join(reasons),
            "confidence": None,
            "entropy": None,
            "margin": None,
        }

    probs = normalize_probabilities(probabilities)
    confidence = float(probs.max())
    entropy = predictive_entropy(probs)
    margin = top2_margin(probs)

    if confidence < float(thresholds["min_confidence"]):
        reasons.append("low_confidence")
    if entropy > float(thresholds["max_entropy"]):
        reasons.append("high_entropy")
    if margin < float(thresholds["min_top2_margin"]):
        reasons.append("low_top2_margin")

    return {
        "manual_review": bool(reasons),
        "reason": ",".join(reasons) if reasons else "confidence_above_threshold",
        "confidence": round(confidence, 6),
        "entropy": round(entropy, 6),
        "margin": round(margin, 6),
    }
