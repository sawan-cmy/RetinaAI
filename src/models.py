from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .constants import DR_CLASSES
from .preprocessing import preprocess_image

CNN_MODEL_SPECS: dict[str, dict[str, Any]] = {
    "simple_cnn": {
        "display_name": "Simple CNN",
        "keras_name": None,
        "preprocess_module": None,
        "input_size": 224,
        "last_conv_layer": "simple_conv_4",
    },
    "efficientnet_b0": {
        "display_name": "EfficientNet-B0",
        "keras_name": "EfficientNetB0",
        "preprocess_module": "efficientnet",
        "input_size": 224,
        "last_conv_layer": "top_conv",
    },
    "efficientnet_b3": {
        "display_name": "EfficientNet-B3",
        "keras_name": "EfficientNetB3",
        "preprocess_module": "efficientnet",
        "input_size": 300,
        "last_conv_layer": "top_conv",
    },
    "resnet50": {
        "display_name": "ResNet50",
        "keras_name": "ResNet50",
        "preprocess_module": "resnet50",
        "input_size": 224,
        "last_conv_layer": "conv5_block3_out",
    },
}

MODEL_ALIASES = {
    "simple-cnn": "simple_cnn",
    "simplecnn": "simple_cnn",
    "efficientnet-b0": "efficientnet_b0",
    "efficientnetb0": "efficientnet_b0",
    "efficientnet-b3": "efficientnet_b3",
    "efficientnetb3": "efficientnet_b3",
    "resnet-50": "resnet50",
    "resnet_50": "resnet50",
}


def canonical_model_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    key = MODEL_ALIASES.get(key, key)
    if key not in CNN_MODEL_SPECS:
        raise ValueError(f"unsupported CNN model {name!r}; choose one of {sorted(CNN_MODEL_SPECS)}")
    return key


def set_reproducible_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass


def make_baseline_model(n_estimators: int = 300, max_depth=None, seed: int = 42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,  # ponytail: sandbox-safe; set to -1 locally when parallel worker pipes are allowed.
    )


def train_baseline(feature_matrix, labels, **kwargs) -> tuple[dict, float]:
    model = make_baseline_model(**kwargs)
    started = perf_counter()
    model.fit(np.asarray(feature_matrix), np.asarray(labels, dtype=int))
    elapsed = perf_counter() - started
    return {"kind": "sklearn_random_forest", "model": model, "classes": list(DR_CLASSES)}, elapsed


def save_model(bundle: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return path


def _metadata_path(model_path: Path) -> Path:
    return model_path / "metadata.json" if model_path.is_dir() else model_path.with_suffix(".json")


def save_keras_model(model, path: str | Path, metadata: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    _metadata_path(path).write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    return path


def _require_torch():
    try:
        import torch
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError("PyTorch and torchvision are required to load .pt CNN model artifacts.") from exc
    return torch, models


def _build_torch_model(arch: str, num_classes: int = 5):
    torch, models = _require_torch()
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        input_size = 224
    else:
        raise ValueError(f"unsupported PyTorch CNN architecture {arch!r}")
    return model, input_size


def _load_torch_model(path: Path) -> dict:
    torch, _ = _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"unsupported PyTorch checkpoint: {path}")
    arch = str(checkpoint.get("arch") or "efficientnet_b0")
    model_name = str(checkpoint.get("model_name") or f"{arch}_torch")
    model, input_size = _build_torch_model(arch, num_classes=len(DR_CLASSES))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return {
        "kind": "torch_cnn",
        "model": model,
        "classes": list(DR_CLASSES),
        "metadata": {
            "model_name": model_name,
            "arch": arch,
            "input_size": input_size,
            "device": str(device),
            "metrics": checkpoint.get("metrics"),
        },
    }


def load_model(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".keras", ".h5"} or path.is_dir():
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to load CNN model artifacts.") from exc

        model = tf.keras.models.load_model(path)
        metadata_path = _metadata_path(path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        model_name = canonical_model_name(metadata.get("model_name", "efficientnet_b0"))
        spec = CNN_MODEL_SPECS[model_name]
        return {
            "kind": "keras_cnn",
            "model": model,
            "classes": list(DR_CLASSES),
            "metadata": {
                "model_name": model_name,
                "input_size": int(metadata.get("input_size", spec["input_size"])),
                "last_conv_layer": metadata.get("last_conv_layer", spec["last_conv_layer"]),
                **metadata,
            },
        }

    if path.suffix.lower() == ".pt":
        return _load_torch_model(path)

    with path.open("rb") as handle:
        bundle = pickle.load(handle)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"unsupported model bundle: {path}")
    return bundle


def predict_probabilities(bundle: dict, feature_vector) -> np.ndarray:
    if bundle.get("kind") in {"keras_cnn", "torch_cnn"}:
        return predict_image_probabilities(bundle, feature_vector)

    model = bundle["model"]
    raw = model.predict_proba(np.asarray(feature_vector, dtype=np.float32).reshape(1, -1))[0]
    classes = getattr(model, "classes_", list(range(len(raw))))
    probabilities = np.zeros(len(DR_CLASSES), dtype=np.float64)
    for class_id, probability in zip(classes, raw):
        probabilities[int(class_id)] = float(probability)
    total = probabilities.sum()
    return probabilities / total if total > 0 else probabilities


def _tf_applications():
    try:
        from tensorflow.keras import applications
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for transfer-learning models.") from exc
    return applications


def get_preprocess_function(model_name: str):
    canonical = canonical_model_name(model_name)
    if canonical == "simple_cnn":
        return lambda image: image / 255.0
    applications = _tf_applications()
    module = getattr(applications, CNN_MODEL_SPECS[canonical]["preprocess_module"])
    return module.preprocess_input


def build_simple_cnn_model(
    num_classes: int = 5,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    learning_rate: float = 1e-4,
    dropout: float = 0.30,
):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for simple CNN training.") from exc

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="simple_conv_1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="simple_bn_1")(x)
    x = tf.keras.layers.MaxPooling2D(name="simple_pool_1")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="simple_conv_2")(x)
    x = tf.keras.layers.BatchNormalization(name="simple_bn_2")(x)
    x = tf.keras.layers.MaxPooling2D(name="simple_pool_2")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="simple_conv_3")(x)
    x = tf.keras.layers.BatchNormalization(name="simple_bn_3")(x)
    x = tf.keras.layers.MaxPooling2D(name="simple_pool_3")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="simple_conv_4")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="severity")(x)
    model = tf.keras.Model(inputs, outputs, name="simple_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_transfer_learning_model(
    model_name: str,
    num_classes: int = 5,
    input_shape: tuple[int, int, int] | None = None,
    learning_rate: float = 1e-4,
    dropout: float = 0.30,
    train_base: bool = False,
    weights: str | None = "imagenet",
):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for EfficientNet/ResNet transfer learning.") from exc

    canonical = canonical_model_name(model_name)
    spec = CNN_MODEL_SPECS[canonical]
    size = int(spec["input_size"])
    input_shape = input_shape or (size, size, 3)
    if canonical == "simple_cnn":
        return build_simple_cnn_model(num_classes, input_shape, learning_rate, dropout)
    base_cls = getattr(tf.keras.applications, spec["keras_name"])
    base = base_cls(weights=weights, include_top=False, input_shape=input_shape)
    base.trainable = train_base

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="severity")(x)
    model = tf.keras.Model(inputs, outputs, name=canonical)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def preprocess_for_model(image_or_path, model_name: str, input_size: int | None = None) -> np.ndarray:
    canonical = canonical_model_name(model_name)
    size = input_size or int(CNN_MODEL_SPECS[canonical]["input_size"])
    image = preprocess_image(image_or_path, size=size).astype("float32")
    batch = np.expand_dims(image, axis=0)
    return get_preprocess_function(canonical)(batch)


def preprocess_for_torch_model(image_or_path, input_size: int = 224):
    torch, _ = _require_torch()
    image = preprocess_image(image_or_path, size=input_size).astype("float32") / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)


def predict_torch_image_probabilities(bundle: dict, image_or_path) -> np.ndarray:
    torch, _ = _require_torch()
    metadata = bundle.get("metadata", {})
    device = next(bundle["model"].parameters()).device
    batch = preprocess_for_torch_model(image_or_path, int(metadata.get("input_size", 224))).to(device)
    with torch.no_grad():
        logits = bundle["model"](batch)
        raw = torch.softmax(logits.float(), dim=1)[0].detach().cpu().numpy()
    probabilities = np.zeros(len(DR_CLASSES), dtype=np.float64)
    probabilities[: min(len(probabilities), len(raw))] = raw[: len(probabilities)]
    total = probabilities.sum()
    return probabilities / total if total > 0 else probabilities


def predict_image_probabilities(bundle: dict, image_or_path) -> np.ndarray:
    if bundle.get("kind") == "torch_cnn":
        return predict_torch_image_probabilities(bundle, image_or_path)

    if bundle.get("kind") != "keras_cnn":
        features = image_or_path
        return predict_probabilities(bundle, features)

    metadata = bundle.get("metadata", {})
    model_name = metadata.get("model_name", "efficientnet_b0")
    batch = preprocess_for_model(image_or_path, model_name, metadata.get("input_size"))
    raw = np.asarray(bundle["model"].predict(batch, verbose=0)[0], dtype=np.float64)
    probabilities = np.zeros(len(DR_CLASSES), dtype=np.float64)
    probabilities[: min(len(probabilities), len(raw))] = raw[: len(probabilities)]
    total = probabilities.sum()
    return probabilities / total if total > 0 else probabilities