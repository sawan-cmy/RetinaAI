from __future__ import annotations

import json
import os
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from src.inference import screen_retina_image
from src.models import save_keras_model


def main() -> int:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(f"TensorFlow is not installed: {exc}") from exc

    tf.keras.utils.set_random_seed(42)
    out_dir = Path("models/smoke")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "efficientnet_b0_smoke.keras"

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", name="smoke_conv")(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax", dtype="float32", name="severity")(x)
    model = tf.keras.Model(inputs, outputs, name="efficientnet_b0_smoke")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    x_train = np.random.default_rng(42).normal(size=(5, 224, 224, 3)).astype("float32")
    y_train = np.arange(5, dtype="int32")
    model.fit(x_train, y_train, epochs=1, verbose=0)

    save_keras_model(
        model,
        model_path,
        {
            "kind": "keras_cnn",
            "model_name": "efficientnet_b0",
            "display_name": "EfficientNet-B0 smoke CNN",
            "input_size": 224,
            "last_conv_layer": "smoke_conv",
            "classes": list(range(5)),
            "seed": 42,
            "smoke_only": True,
        },
    )

    result = screen_retina_image(
        "tests/_self_check/synthetic_retina.png",
        model_path=model_path,
        fallback_model_path=None,
        output_dir="reports/smoke_cnn",
        patient_id="smoke-demo",
        site_id="smoke",
    )
    assert result["prediction"]["status"] == "available", result["prediction"]
    assert result["model"]["fallback_mode"] is False, result["model"]
    assert Path(result["outputs"]["gradcam_path"]).exists(), result["outputs"]
    assert Path(result["outputs"]["report_path"]).exists(), result["outputs"]

    summary_path = Path("reports/smoke_cnn/summary.json")
    summary_path.write_text(json.dumps({"model_path": str(model_path), "result": result}, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "model_path": str(model_path), "summary_path": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
