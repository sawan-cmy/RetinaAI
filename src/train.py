from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import yaml
from sklearn.utils.class_weight import compute_class_weight

from .artifact_package import package_training_run
from .calibration import write_calibration_artifacts
from .cards import write_dataset_card, write_model_card
from .datasets import DATASET_SPECS, class_distribution, load_labels, split_dataframe
from .evaluate import calculate_metrics, plot_confusion_matrix, save_metrics
from .models import (
    CNN_MODEL_SPECS,
    build_transfer_learning_model,
    canonical_model_name,
    get_preprocess_function,
    predict_probabilities,
    save_keras_model,
    save_model,
    set_reproducible_seed,
    train_baseline,
)
from .preprocessing import extract_handcrafted_features


def _features(paths) -> np.ndarray:
    return np.vstack([extract_handcrafted_features(path) for path in paths])


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if not path or not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _class_weights(labels: np.ndarray) -> dict[int, float]:
    classes = np.unique(labels.astype(int))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels.astype(int))
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for CNN training. Install requirements-deep-learning.txt.") from exc
    return tf


def _make_tf_dataset(df, model_name: str, batch_size: int, shuffle: bool, seed: int):
    tf = _require_tensorflow()
    canonical = canonical_model_name(model_name)
    size = int(CNN_MODEL_SPECS[canonical]["input_size"])
    preprocess_fn = get_preprocess_function(canonical)
    paths = df["path"].astype(str).to_numpy()
    labels = df["label"].astype(int).to_numpy()

    def _load(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [size, size])
        image = tf.cast(image, tf.float32)
        image = preprocess_fn(image)
        return image, tf.cast(label, tf.int32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=max(1, len(df)), seed=seed, reshuffle_each_iteration=True)
    return dataset.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _finalize_artifacts(
    model_name: str,
    model_path: str | Path,
    metrics: dict,
    y_true,
    probabilities,
    dataset_name: str,
    labels_csv: str | Path | None,
    image_dir: str | Path | None,
    label_mapping: dict | None,
    checkpoint_path: str | Path | None = None,
    deployment_site: str | None = None,
) -> dict:
    site = deployment_site or dataset_name
    calibration = write_calibration_artifacts(y_true, probabilities, site=site, model_name=model_name)
    metrics["calibration_artifacts"] = calibration
    distributions = {
        "train": metrics.get("train_distribution"),
        "validation": metrics.get("validation_distribution"),
        "test": metrics.get("test_distribution"),
    }
    dataset_card = write_dataset_card(dataset_name, labels_csv, image_dir, distributions, label_mapping)
    model_card = write_model_card(model_name, model_path, metrics, dataset_name, checkpoint_path=checkpoint_path, calibration_artifacts=calibration)
    metrics["dataset_card_path"] = str(dataset_card)
    metrics["model_card_path"] = str(model_card)
    metrics_path = save_metrics(metrics, f"reports/metrics_{model_name}.json")
    metrics["metrics_path"] = str(metrics_path)
    metrics["package_path"] = str(package_training_run(model_name))
    save_metrics(metrics, metrics_path)
    return metrics


def train_cnn_from_dataframe(
    train_df,
    val_df,
    test_df,
    model_name: str,
    out_dir: str | Path = "models",
    epochs: int = 20,
    batch_size: int = 32,
    seed: int = 42,
    early_stopping_patience: int = 5,
    learning_rate: float = 1e-4,
    mixed_precision: bool = True,
    train_base: bool = False,
    dataset_name: str = "unknown",
    labels_csv: str | Path | None = None,
    image_dir: str | Path | None = None,
    label_mapping: dict | None = None,
    deployment_site: str | None = None,
) -> dict:
    tf = _require_tensorflow()
    canonical = canonical_model_name(model_name)
    set_reproducible_seed(seed)

    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    model = build_transfer_learning_model(
        canonical,
        num_classes=5,
        learning_rate=learning_rate,
        train_base=train_base,
    )
    train_ds = _make_tf_dataset(train_df, canonical, batch_size, shuffle=True, seed=seed)
    val_ds = _make_tf_dataset(val_df, canonical, batch_size, shuffle=False, seed=seed)
    test_ds = _make_tf_dataset(test_df, canonical, batch_size, shuffle=False, seed=seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{canonical}_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(1, early_stopping_patience // 2)),
    ]

    labels = train_df["label"].to_numpy(dtype=int)
    started = perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=_class_weights(labels),
        callbacks=callbacks,
        verbose=1,
    )
    train_seconds = perf_counter() - started

    y_true = np.concatenate([batch_labels.numpy() for _, batch_labels in test_ds])
    probabilities = np.asarray(model.predict(test_ds.map(lambda images, _: images), verbose=0), dtype=float)
    metrics = calculate_metrics(y_true, probabilities)
    metrics["train_seconds"] = round(float(train_seconds), 3)
    metrics["train_distribution"] = class_distribution(train_df)
    metrics["validation_distribution"] = class_distribution(val_df)
    metrics["test_distribution"] = class_distribution(test_df)

    spec = CNN_MODEL_SPECS[canonical]
    metadata = {
        "kind": "keras_cnn",
        "model_name": canonical,
        "display_name": spec["display_name"],
        "input_size": spec["input_size"],
        "last_conv_layer": spec["last_conv_layer"],
        "classes": list(range(5)),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "mixed_precision": mixed_precision,
    }
    final_path = save_keras_model(model, out_dir / f"{canonical}.keras", metadata)
    if checkpoint_path.exists():
        checkpoint_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / f"{canonical}_history.json").write_text(json.dumps(history.history, indent=2, default=float), encoding="utf-8")
    metrics["model_path"] = str(final_path)
    metrics["checkpoint_path"] = str(checkpoint_path)
    return _finalize_artifacts(
        canonical,
        final_path,
        metrics,
        y_true,
        probabilities,
        dataset_name,
        labels_csv,
        image_dir,
        label_mapping,
        checkpoint_path=checkpoint_path,
        deployment_site=deployment_site,
    )


def train_from_dataset(
    labels_csv: str | Path,
    image_dir: str | Path,
    out_dir: str | Path = "models",
    max_images: int | None = None,
    dataset: str = "aptos",
    model_name: str = "baseline_sklearn",
    config_path: str | Path | None = "configs/train.yaml",
) -> dict:
    config = _load_yaml(config_path)
    split_cfg = config.get("split", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    dataset_cfg = config.get("dataset", {})

    seed = int(split_cfg.get("seed", training_cfg.get("seed", 42)))
    set_reproducible_seed(seed)
    label_mapping = model_cfg.get("label_mapping") or dataset_cfg.get("label_mapping")
    if label_mapping is None:
        label_mapping = dict(DATASET_SPECS.get(dataset.lower(), DATASET_SPECS["aptos"]).label_mapping)
    df = load_labels(labels_csv, image_dir, dataset=dataset, label_mapping=label_mapping)
    if max_images:
        df = df.sample(n=min(max_images, len(df)), random_state=seed).reset_index(drop=True)

    train_df, val_df, test_df = split_dataframe(
        df,
        train_size=float(split_cfg.get("train", 0.70)),
        val_size=float(split_cfg.get("validation", 0.15)),
        seed=seed,
    )

    deployment_site = str(dataset_cfg.get("deployment_site") or dataset)
    selected = model_name.lower()
    if selected in {"baseline", "baseline_sklearn", "random_forest", "rf"}:
        x_train = _features(train_df["path"])
        y_train = train_df["label"].to_numpy(dtype=int)
        x_test = _features(test_df["path"])
        y_test = test_df["label"].to_numpy(dtype=int)
        bundle, train_seconds = train_baseline(
            x_train,
            y_train,
            n_estimators=int(model_cfg.get("n_estimators", 300)),
            max_depth=model_cfg.get("max_depth"),
            seed=seed,
        )
        probabilities = np.vstack([predict_probabilities(bundle, row) for row in x_test])
        metrics = calculate_metrics(y_test, probabilities)
        metrics["train_seconds"] = round(float(train_seconds), 3)
        metrics["train_distribution"] = class_distribution(train_df)
        metrics["validation_distribution"] = class_distribution(val_df)
        metrics["test_distribution"] = class_distribution(test_df)
        out_dir = Path(out_dir)
        model_path = save_model(bundle, out_dir / "baseline_sklearn.pkl")
        plot_confusion_matrix(metrics["confusion_matrix"], "reports/figures/confusion_matrix_baseline_sklearn.png")
        metrics["model_path"] = str(model_path)
        return _finalize_artifacts(
            "baseline_sklearn",
            model_path,
            metrics,
            y_test,
            probabilities,
            dataset,
            labels_csv,
            image_dir,
            label_mapping,
            deployment_site=deployment_site,
        )

    return train_cnn_from_dataframe(
        train_df,
        val_df,
        test_df,
        model_name=selected,
        out_dir=out_dir,
        epochs=int(training_cfg.get("epochs", 20)),
        batch_size=int(training_cfg.get("batch_size", 32)),
        seed=seed,
        early_stopping_patience=int(training_cfg.get("early_stopping_patience", 5)),
        learning_rate=float(training_cfg.get("learning_rate", 1e-4)),
        mixed_precision=bool(training_cfg.get("mixed_precision", True)),
        train_base=bool(model_cfg.get("train_base", False)),
        dataset_name=dataset,
        labels_csv=labels_csv,
        image_dir=image_dir,
        label_mapping=label_mapping,
        deployment_site=deployment_site,
    )


def train_from_aptos(labels_csv: str | Path, image_dir: str | Path, out_dir: str | Path = "models", max_images: int | None = None) -> dict:
    return train_from_dataset(labels_csv, image_dir, out_dir=out_dir, max_images=max_images, dataset="aptos", model_name="baseline_sklearn")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train RetinaAI baseline or transfer-learning CNN models.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--labels-csv", default="data/raw/aptos2019/train.csv")
    parser.add_argument("--image-dir", default="data/raw/aptos2019/images_288_scaled")
    parser.add_argument("--dataset", default="aptos")
    parser.add_argument("--model", default=None, help="baseline_sklearn, simple_cnn, efficientnet_b0, efficientnet_b3, or resnet50")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    selected_model = args.model or cfg.get("model", {}).get("name") or cfg.get("model", {}).get("baseline", "baseline_sklearn")
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    metrics = train_from_dataset(
        args.labels_csv,
        args.image_dir,
        args.out_dir,
        args.max_images,
        dataset=args.dataset,
        model_name=selected_model,
        config_path=args.config,
    )
    print(yaml.safe_dump(metrics, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())