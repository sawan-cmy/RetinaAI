from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    image_id_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    label_mapping: Mapping[int | str, int] = field(default_factory=dict)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "aptos": DatasetSpec("aptos", ("id_code", "image", "image_id"), ("diagnosis", "label"), {}),
    "aptos2019": DatasetSpec("aptos2019", ("id_code", "image", "image_id"), ("diagnosis", "label"), {}),
    "eyepacs": DatasetSpec("eyepacs", ("image", "image_id", "id_code"), ("level", "diagnosis", "label"), {}),
    "messidor": DatasetSpec("messidor", ("image", "image_id", "id_code"), ("diagnosis", "grade", "label"), {}),
    "idrid": DatasetSpec("idrid", ("image", "image_id", "id_code"), ("diagnosis", "grade", "label"), {}),
}


def resolve_image_path(image_id: str, image_dir: str | Path) -> Path | None:
    image_dir = Path(image_dir)
    candidate = image_dir / image_id
    if candidate.suffix and candidate.exists():
        return candidate
    for suffix in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _first_existing(columns: Sequence[str], candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"could not infer {label} column. Available columns: {list(columns)}")


def _normalize_mapping(mapping: Mapping[int | str, int] | None) -> dict[int | str, int]:
    out: dict[int | str, int] = {}
    for key, value in (mapping or {}).items():
        try:
            out[int(key)] = int(value)
        except (TypeError, ValueError):
            out[str(key)] = int(value)
    return out


def map_label(value, mapping: Mapping[int | str, int] | None = None) -> int:
    normalized = _normalize_mapping(mapping)
    if normalized:
        if value in normalized:
            return normalized[value]
        try:
            int_value = int(value)
            if int_value in normalized:
                return normalized[int_value]
        except (TypeError, ValueError):
            pass
        text_value = str(value)
        if text_value in normalized:
            return normalized[text_value]
        raise ValueError(f"label {value!r} is missing from label mapping")
    return int(value)


def load_labels(
    csv_path: str | Path,
    image_dir: str | Path,
    dataset: str = "aptos",
    image_id_col: str | None = None,
    label_col: str | None = None,
    label_mapping: Mapping[int | str, int] | None = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    spec = DATASET_SPECS.get(dataset.lower(), DATASET_SPECS["aptos"])
    df = pd.read_csv(csv_path)
    image_id_col = image_id_col or _first_existing(df.columns, spec.image_id_columns, "image id")
    label_col = label_col or _first_existing(df.columns, spec.label_columns, "label")
    mapping = label_mapping if label_mapping is not None else spec.label_mapping

    rows = []
    for row in df.itertuples(index=False):
        values = row._asdict()
        image_id = str(values[image_id_col])
        path = resolve_image_path(image_id, image_dir)
        if path is not None:
            rows.append(
                {
                    "dataset": spec.name,
                    "image_id": image_id,
                    "path": str(path),
                    "label": map_label(values[label_col], mapping),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"no images from {csv_path} were found in {image_dir}")
    return out


def load_aptos_labels(csv_path: str | Path, image_dir: str | Path) -> pd.DataFrame:
    return load_labels(csv_path, image_dir, dataset="aptos")


def class_distribution(df: pd.DataFrame, label_col: str = "label") -> dict[int, int]:
    return {int(label): int(count) for label, count in df[label_col].value_counts().sort_index().items()}


def _stratify_or_none(df: pd.DataFrame, label_col: str):
    counts = df[label_col].value_counts()
    return df[label_col] if len(counts) > 1 and int(counts.min()) >= 2 else None


def split_dataframe(
    df: pd.DataFrame,
    label_col: str = "label",
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1")
    if not 0 <= val_size < 1:
        raise ValueError("val_size must be between 0 and 1")
    if train_size + val_size >= 1:
        raise ValueError("train_size + val_size must leave a test split")

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=_stratify_or_none(df, label_col),
    )
    relative_val = val_size / (1 - train_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val,
        random_state=seed,
        stratify=_stratify_or_none(temp_df, label_col),
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)