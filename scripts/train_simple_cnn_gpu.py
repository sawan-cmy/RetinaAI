from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
NUM_CLASSES = 5


class RetinaDataset(Dataset):
    def __init__(self, rows: list[dict], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(row["path"]).convert("RGB")
        return self.transform(image), torch.tensor(row["label"], dtype=torch.long)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 16),
            self._block(16, 32),
            self._block(32, 64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, images):
        return self.classifier(self.features(images))


def resolve_image_path(image_id: str, image_dir: Path) -> Path | None:
    candidate = image_dir / image_id
    if candidate.suffix and candidate.exists():
        return candidate
    for suffix in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_rows(labels_csv: Path, image_dir: Path) -> list[dict]:
    rows: list[dict] = []
    with labels_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for item in reader:
            image_id = item.get("id_code") or item.get("image") or item.get("image_id")
            label = item.get("diagnosis") or item.get("label")
            if image_id is None or label is None:
                continue
            path = resolve_image_path(str(image_id), image_dir)
            if path is not None:
                rows.append({"image_id": str(image_id), "path": str(path), "label": int(label)})
    if not rows:
        raise RuntimeError(f"no images from {labels_csv} were found in {image_dir}")
    return rows


def stratified_split(rows: list[dict], seed: int, train_size: float = 0.70, val_size: float = 0.15):
    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[int(row["label"])].append(row)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
        n = len(label_rows)
        train_end = int(round(n * train_size))
        val_end = train_end + int(round(n * val_size))
        train.extend(label_rows[:train_end])
        val.extend(label_rows[train_end:val_end])
        test.extend(label_rows[val_end:])
    for split in (train, val, test):
        rng.shuffle(split)
    return train, val, test


def distribution(rows: list[dict]) -> dict[int, int]:
    counts = Counter(int(row["label"]) for row in rows)
    return {label: int(counts.get(label, 0)) for label in range(NUM_CLASSES)}


def class_weights(rows: list[dict], device) -> torch.Tensor:
    counts = distribution(rows)
    total = sum(counts.values())
    weights = [total / (NUM_CLASSES * max(1, counts[label])) for label in range(NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loaders(train_rows, val_rows, test_rows, batch_size: int, workers: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    return (
        DataLoader(RetinaDataset(train_rows, train_transform), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True),
        DataLoader(RetinaDataset(val_rows, eval_transform), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
        DataLoader(RetinaDataset(test_rows, eval_transform), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
    )


def run_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += labels.size(0)
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    probabilities: list[list[float]] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        probs = torch.softmax(logits.float(), dim=1)
        total_loss += float(loss.detach()) * labels.size(0)
        correct += int((probs.argmax(dim=1) == labels).sum())
        total += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        probabilities.extend(probs.cpu().tolist())
    return total_loss / max(1, total), correct / max(1, total), y_true, probabilities


def calculate_metrics(y_true: list[int], probabilities: list[list[float]]) -> dict:
    probs = np.asarray(probabilities, dtype=np.float64)
    true = np.asarray(y_true, dtype=np.int64)
    pred = probs.argmax(axis=1)
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for actual, predicted in zip(true, pred):
        matrix[int(actual), int(predicted)] += 1

    precision = []
    recall = []
    f1 = []
    support = []
    for label in range(NUM_CLASSES):
        tp = matrix[label, label]
        fp = matrix[:, label].sum() - tp
        fn = matrix[label, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(float(p))
        recall.append(float(r))
        f1.append(float((2 * p * r / (p + r)) if p + r else 0.0))
        support.append(int(matrix[label, :].sum()))

    positive = true > 0
    false_negative = float(((pred == 0) & positive).sum() / positive.sum()) if positive.any() else None
    weights = np.asarray(support, dtype=np.float64) / max(1, sum(support))
    return {
        "accuracy": float((pred == true).mean()) if len(true) else 0.0,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.sum(np.asarray(f1) * weights)),
        "false_negative_rate_any_dr": false_negative,
        "auc_ovr_macro": None,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a simple CNN on GPU with PyTorch.")
    parser.add_argument("--labels-csv", default="data/raw/aptos2019/train.csv")
    parser.add_argument("--image-dir", default="data/raw/aptos2019/images_288_scaled")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this runner; refusing to train on CPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda")
    started = perf_counter()
    rows = load_rows(Path(args.labels_csv), Path(args.image_dir))
    train_rows, val_rows, test_rows = stratified_split(rows, args.seed)
    train_loader, val_loader, test_loader = make_loaders(train_rows, val_rows, test_rows, args.batch_size, args.workers)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "simple_cnn_torch_best.pt"
    final_path = out_dir / "simple_cnn_torch.pt"

    best_val = float("inf")
    stale_epochs = 0
    history: list[dict] = []
    print(json.dumps({"device": torch.cuda.get_device_name(0), "cuda": True, "train_distribution": distribution(train_rows), "validation_distribution": distribution(val_rows), "test_distribution": distribution(test_rows)}), flush=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_val:
            best_val = val_loss
            stale_epochs = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "seed": args.seed, "model_name": "simple_cnn_torch"}, best_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc, y_true, probabilities = evaluate(model, test_loader, criterion, device)
    metrics = calculate_metrics(y_true, probabilities)
    metrics.update(
        {
            "model_name": "simple_cnn_torch",
            "device": torch.cuda.get_device_name(0),
            "cuda": True,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "train_seconds": round(perf_counter() - started, 3),
            "train_distribution": distribution(train_rows),
            "validation_distribution": distribution(val_rows),
            "test_distribution": distribution(test_rows),
            "model_path": str(final_path),
            "checkpoint_path": str(best_path),
        }
    )
    torch.save({"model_state": model.state_dict(), "seed": args.seed, "model_name": "simple_cnn_torch", "metrics": metrics}, final_path)
    (out_dir / "simple_cnn_torch_history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
    metrics_path = reports_dir / "metrics_simple_cnn_torch.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"ok": True, "metrics_path": str(metrics_path), "model_path": str(final_path), "checkpoint_path": str(best_path), "metrics": metrics}, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())