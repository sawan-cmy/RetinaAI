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
from torchvision import models, transforms

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
NUM_CLASSES = 5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
    # ponytail: sqrt weighting keeps minority classes visible without letting tiny classes dominate the first baseline.
    weights = [(total / (NUM_CLASSES * max(1, counts[label]))) ** 0.5 for label in range(NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def make_loaders(train_rows, val_rows, test_rows, batch_size: int, workers: int, image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return (
        DataLoader(RetinaDataset(train_rows, train_transform), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True),
        DataLoader(RetinaDataset(val_rows, eval_transform), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
        DataLoader(RetinaDataset(test_rows, eval_transform), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True),
    )


def build_model(arch: str) -> tuple[nn.Module, list[nn.Parameter], list[nn.Parameter], int]:
    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
        head = list(model.classifier.parameters())
        image_size = 224
    elif arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)
        head = list(model.fc.parameters())
        image_size = 224
    elif arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
        head = list(model.classifier.parameters())
        image_size = 224
    else:
        raise ValueError("arch must be efficientnet_b0, resnet18, or mobilenet_v3_small")

    head_ids = {id(param) for param in head}
    backbone = [param for param in model.parameters() if id(param) not in head_ids]
    return model, backbone, head, image_size


def freeze_backbone(backbone: list[nn.Parameter], freeze: bool) -> None:
    for param in backbone:
        param.requires_grad = not freeze


def make_optimizer(backbone: list[nn.Parameter], head: list[nn.Parameter], lr: float, backbone_lr: float):
    return torch.optim.AdamW(
        [
            {"params": [p for p in backbone if p.requires_grad], "lr": backbone_lr},
            {"params": head, "lr": lr},
        ],
        weight_decay=1e-4,
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
    parser = argparse.ArgumentParser(description="GPU transfer-learning trainer for RetinaAI.")
    parser.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "resnet18", "mobilenet_v3_small"])
    parser.add_argument("--labels-csv", default="data/raw/aptos2019/train.csv")
    parser.add_argument("--image-dir", default="data/raw/aptos2019/images_288_scaled")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--selection-metric", default="val_loss", choices=["val_loss", "val_accuracy"])
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
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
    model, backbone, head, image_size = build_model(args.arch)
    model = model.to(device)
    freeze_backbone(backbone, True)
    train_loader, val_loader, test_loader = make_loaders(train_rows, val_rows, test_rows, args.batch_size, args.workers, image_size)

    criterion = nn.CrossEntropyLoss(weight=class_weights(train_rows, device), label_smoothing=0.03)
    optimizer = make_optimizer(backbone, head, args.lr, args.backbone_lr)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.run_name or f"{args.arch}_torch_transfer"
    best_path = out_dir / f"{model_name}_best.pt"
    final_path = out_dir / f"{model_name}.pt"

    best_score = float("inf") if args.selection_metric == "val_loss" else -float("inf")
    stale_epochs = 0
    history: list[dict] = []
    print(json.dumps({"device": torch.cuda.get_device_name(0), "cuda": True, "arch": args.arch, "pretrained": True, "train_distribution": distribution(train_rows), "validation_distribution": distribution(val_rows), "test_distribution": distribution(test_rows)}), flush=True)

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            freeze_backbone(backbone, False)
            optimizer = make_optimizer(backbone, head, args.lr * 0.5, args.backbone_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch + 1))
            print(json.dumps({"event": "unfreeze_backbone", "epoch": epoch}), flush=True)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc, "val_loss": val_loss, "val_accuracy": val_acc}
        history.append(row)
        print(json.dumps(row), flush=True)
        score = val_loss if args.selection_metric == "val_loss" else val_acc
        improved = score < best_score if args.selection_metric == "val_loss" else score > best_score
        if improved:
            best_score = score
            stale_epochs = 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "seed": args.seed, "arch": args.arch, "model_name": model_name, "selection_metric": args.selection_metric}, best_path)
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
            "model_name": model_name,
            "arch": args.arch,
            "pretrained": True,
            "selection_metric": args.selection_metric,
            "device": torch.cuda.get_device_name(0),
            "cuda": True,
            "best_epoch": int(checkpoint["epoch"]),
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
    torch.save({"model_state": model.state_dict(), "seed": args.seed, "arch": args.arch, "model_name": model_name, "metrics": metrics}, final_path)
    (out_dir / f"{model_name}_history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
    metrics_path = reports_dir / f"metrics_{model_name}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"ok": True, "metrics_path": str(metrics_path), "model_path": str(final_path), "checkpoint_path": str(best_path), "metrics": metrics}, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())