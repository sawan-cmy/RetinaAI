from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .preprocessing import load_image


class GradCamUnavailable(RuntimeError):
    pass


def save_unavailable_explanation(image_or_path, output_path: str | Path, message: str = "Grad-CAM unavailable: train a CNN model first") -> Path:
    image = load_image(image_or_path).copy()
    image = cv2.resize(image, (640, 420), interpolation=cv2.INTER_AREA)
    overlay = image.copy()
    cv2.rectangle(overlay, (18, 18), (622, 92), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.55, image, 0.45, 0)
    cv2.putText(image, message[:76], (32, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return output_path


def overlay_heatmap(image_or_path, heatmap: np.ndarray, output_path: str | Path, alpha: float = 0.38) -> Path:
    image = cv2.resize(load_image(image_or_path), (640, 420), interpolation=cv2.INTER_AREA)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be a 2D array")
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmap = np.maximum(heatmap, 0)
    if float(heatmap.max()) > 0:
        heatmap = heatmap / float(heatmap.max())
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 1 - alpha, colored, alpha, 0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return output_path


def _nested_layer(model, name: str):
    try:
        return model.get_layer(name)
    except Exception:
        pass
    for layer in getattr(model, "layers", []):
        if hasattr(layer, "get_layer"):
            try:
                return layer.get_layer(name)
            except Exception:
                continue
    raise GradCamUnavailable(f"Model has no layer named {name!r}.")


def find_last_conv_layer(model) -> str:
    for layer in reversed(getattr(model, "layers", [])):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None and hasattr(layer, "output"):
            output_shape = getattr(layer.output, "shape", None)
        if output_shape is not None and len(output_shape) == 4:
            return layer.name
        if hasattr(layer, "layers"):
            try:
                return find_last_conv_layer(layer)
            except GradCamUnavailable:
                pass
    raise GradCamUnavailable("Could not find a convolutional feature layer for Grad-CAM.")


def generate_keras_gradcam(
    model,
    image_or_path,
    last_conv_layer_name: str | None,
    output_path: str | Path,
    class_index: int | None = None,
    size: int = 224,
    model_name: str = "efficientnet_b0",
) -> Path:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise GradCamUnavailable("TensorFlow is not installed, so CNN Grad-CAM cannot run.") from exc

    from .models import preprocess_for_model

    last_conv_layer_name = last_conv_layer_name or find_last_conv_layer(model)
    conv_layer = _nested_layer(model, last_conv_layer_name)
    batch = preprocess_for_model(image_or_path, model_name, size)

    grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(batch)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    gradients = tape.gradient(class_score, conv_outputs)
    if gradients is None:
        raise GradCamUnavailable("Could not compute gradients for Grad-CAM.")
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1).numpy()
    return overlay_heatmap(image_or_path, heatmap, output_path)