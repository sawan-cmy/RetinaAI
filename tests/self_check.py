from __future__ import annotations

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference import screen_retina_image
from src.quality_check import assess_quality
from src.uncertainty import route_case


def make_synthetic_retina(path: Path) -> Path:
    image = np.zeros((360, 360, 3), dtype=np.uint8)
    cv2.circle(image, (180, 180), 145, (92, 42, 25), -1)
    cv2.circle(image, (132, 168), 26, (210, 165, 120), -1)
    for offset in range(-90, 100, 30):
        cv2.line(image, (132, 168), (260, 180 + offset), (160, 55, 45), 3)
        cv2.line(image, (132, 168), (70, 170 + offset // 2), (145, 45, 38), 2)
    noise = np.random.default_rng(42).normal(0, 6, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)
    return path


def main() -> int:
    tmp = ROOT / "tests" / "_self_check"
    tmp.mkdir(parents=True, exist_ok=True)
    good = make_synthetic_retina(tmp / "synthetic_retina.png")
    dark = tmp / "dark.png"
    Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(dark)

    good_quality = assess_quality(good)
    dark_quality = assess_quality(dark)
    assert good_quality.status == "accepted", good_quality
    assert dark_quality.status == "rejected", dark_quality

    confident = route_case("accepted", [0.01, 0.02, 0.92, 0.03, 0.02])
    missing = route_case("accepted", None)
    assert confident["manual_review"] is False, confident
    assert missing["manual_review"] is True and missing["reason"] == "model_missing", missing

    result = screen_retina_image(good, model_path=tmp / "missing.pkl", fallback_model_path=None, output_dir=tmp / "reports")
    assert result["quality"]["status"] == "accepted", result
    assert result["prediction"]["status"] == "model_missing", result
    assert result["uncertainty"]["manual_review"] is True, result
    assert Path(result["outputs"]["report_path"]).exists(), result
    assert Path(result["outputs"]["explanation_path"]).exists(), result

    corrupt_model = tmp / "corrupt.pkl"
    with corrupt_model.open("wb") as handle:
        pickle.dump({"not_model": True}, handle)
    corrupt = screen_retina_image(good, model_path=corrupt_model, output_dir=tmp / "corrupt_reports")
    assert corrupt["prediction"]["status"] == "model_error", corrupt
    assert "model_error" in corrupt["uncertainty"]["reason"], corrupt

    default_model = ROOT / "models" / "baseline_sklearn.pkl"
    if default_model.exists():
        model_result = screen_retina_image(good, model_path=default_model, output_dir=tmp / "model_reports")
        assert len(model_result["prediction"].get("probabilities") or []) == 5, model_result
        assert model_result["prediction"]["class_id"] in {0, 1, 2, 3, 4}, model_result

    print("self-check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
