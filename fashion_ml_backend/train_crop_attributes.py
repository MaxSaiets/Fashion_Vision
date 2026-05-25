"""
Train the attribute classifier on cropped image regions.

The available local Fashionpedia attribute annotations are image-level, not
instance-level. This script therefore uses weak supervision: random/center crops
inherit the image attribute vector. It matches the new inference architecture
better than full-image-only training, while still saving resumable checkpoints.
"""
import argparse
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import yaml

from src.losses import compute_pos_weight, get_loss_fn
from src.metrics import multi_label_metrics
from src.models import MultiLabelClassifier


STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\nStop requested by signal {signum}; checkpoint will be saved safely.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def resolve_image_path(base: Path, split: str, file_name: str) -> Path:
    fname = file_name.split("/")[-1]
    candidates = [
        base / split / fname,
        base / f"{split}2020" / fname,
        base / "test" / fname,
        base / split / file_name,
        base / f"{split}2020" / file_name,
        base / file_name,
        base / fname,
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


class WeakFashionpediaCropDataset(Dataset):
    def __init__(
        self,
        split: str,
        annotations_path: str,
        images_dir: str,
        transform,
        num_attributes: int = 294,
        max_samples: int | None = None,
    ):
        self.split = split
        self.images_base = Path(images_dir)
        self.transform = transform
        self.num_attributes = num_attributes

        with open(annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.attr_ids = sorted(a["id"] for a in data.get("attributes", []))[:num_attributes]
        attr_to_idx = {aid: idx for idx, aid in enumerate(self.attr_ids)}
        file_names = {img["id"]: img["file_name"] for img in data.get("images", [])}

        samples = []
        for ann in data.get("annotations", []):
            image_id = ann.get("image_id")
            if image_id not in file_names:
                continue
            labels = np.zeros(num_attributes, dtype=np.float32)
            for attr_id in ann.get("attribute_ids", []):
                idx = attr_to_idx.get(attr_id)
                if idx is not None:
                    labels[idx] = 1.0
            if labels.sum() == 0:
                continue
            image_path = resolve_image_path(self.images_base, split, file_names[image_id])
            if image_path.exists():
                samples.append({"image_path": str(image_path), "labels": labels})
            if max_samples and len(samples) >= max_samples:
                break

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        return self.transform(image), torch.from_numpy(sample["labels"])


def build_transforms(config: dict, is_train: bool):
    resize = config.get("augmentation", {}).get("val", {}).get("resize", [224, 224])
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(resize, scale=(0.35, 1.0), ratio=(0.65, 1.55)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def save_checkpoint(path, model_name, model, optimizer, scheduler, epoch, batch_idx, best_f1, best_epoch, patience_counter):
    atomic_torch_save(
        {
            "model_name": model_name,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
        },
        path,
    )


def run_epoch(model, loader, optimizer, loss_fn, device, epoch, total_epochs, checkpoint_every, checkpoint_path, state):
    model.train()
    total_loss = 0.0
    preds_all, labels_all = [], []
    pbar = tqdm(loader, desc=f"[crop-{state['model_name']}] Epoch {epoch}/{total_epochs}", unit="batch", ncols=100)

    for batch_idx, (images, labels) in enumerate(pbar, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds_all.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels_all.append(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{total_loss / batch_idx:.4f}")

        if checkpoint_every and batch_idx % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, state["model_name"], model, optimizer, state["scheduler"], epoch, batch_idx, **state["best"])
        if STOP_REQUESTED:
            break

    metrics = multi_label_metrics(np.vstack(labels_all), np.vstack(preds_all))
    return total_loss / max(1, len(preds_all)), metrics


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="[crop] Val", unit="batch", leave=False, ncols=80):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, labels).item()
            preds_all.append(torch.sigmoid(logits).cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    metrics = multi_label_metrics(np.vstack(labels_all), np.vstack(preds_all))
    return total_loss / max(1, len(preds_all)), metrics


def main():
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model_name", default="crop_resnet50")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init_from", default="outputs/best_resnet50.pt")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("project", {}).get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.get("project", {}).get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    ds_cfg = config["dataset"]
    train_cfg = config["training"]
    ann_dir = Path(ds_cfg.get("annotations_dir", "data/annotations"))
    images_dir = ds_cfg.get("images_dir", "data/images")
    epochs = args.epochs or train_cfg.get("num_epochs", 15)
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)

    train_ds = WeakFashionpediaCropDataset(
        "train",
        str(ann_dir / "attributes_train2020.json"),
        images_dir,
        build_transforms(config, True),
        max_samples=args.max_samples,
    )
    val_ds = WeakFashionpediaCropDataset(
        "val",
        str(ann_dir / "attributes_val2020.json"),
        images_dir,
        build_transforms(config, False),
        max_samples=args.max_samples,
    )
    print(f"Device: {device} | train={len(train_ds)} val={len(val_ds)} classes={train_ds.num_attributes}")
    print("Crop training mode: weak supervision from image-level Fashionpedia attributes.")

    labels = torch.from_numpy(np.stack([s["labels"] for s in train_ds.samples]))
    pos_weight = compute_pos_weight(labels, train_ds.num_attributes).to(device) if train_cfg.get("use_pos_weight") else None
    loss_fn = get_loss_fn(
        use_focal=train_cfg.get("use_focal_loss", False),
        focal_gamma=train_cfg.get("focal_gamma", 2.0),
        pos_weight=pos_weight,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MultiLabelClassifier(args.backbone, num_classes=train_ds.num_attributes, pretrained=True).to(device)
    if args.init_from and Path(args.init_from).exists() and not args.resume:
        model.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"Initialized from {args.init_from}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("learning_rate", 1e-4)), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(train_cfg.get("min_lr", 1e-6)))

    ckpt_path = output_dir / f"checkpoint_{args.model_name}.pt"
    best_path = output_dir / f"best_{args.model_name}.pt"
    log_path = output_dir / f"training_log_{args.model_name}.jsonl"
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    start_epoch = 1

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_f1 = ckpt.get("best_f1", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        patience_counter = ckpt.get("patience_counter", 0)
        saved_epoch = ckpt.get("epoch", 0)
        saved_batch = ckpt.get("batch_idx", 0)
        start_epoch = saved_epoch if saved_batch else saved_epoch + 1
        print(f"Resuming from epoch {start_epoch}; best F1={best_f1:.4f} @ {best_epoch}")
        if saved_batch:
            print(f"Checkpoint was saved mid-epoch at batch {saved_batch}; repeating epoch {saved_epoch} safely.")

    patience = train_cfg.get("patience", 5)
    min_delta = train_cfg.get("min_delta", 0.001)

    for epoch in range(start_epoch, epochs + 1):
        state = {
            "model_name": args.model_name,
            "scheduler": scheduler,
            "best": {
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
            },
        }
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, loss_fn, device, epoch, epochs, args.checkpoint_every, ckpt_path, state)
        save_checkpoint(ckpt_path, args.model_name, model, optimizer, scheduler, epoch, 0, best_f1, best_epoch, patience_counter)
        if STOP_REQUESTED:
            print("Stopped after safe checkpoint.")
            return

        val_loss, val_metrics = validate(model, val_loader, loss_fn, device)
        scheduler.step()

        improved = val_metrics["f1_micro"] > best_f1 + min_delta
        if improved:
            best_f1 = val_metrics["f1_micro"]
            best_epoch = epoch
            patience_counter = 0
            atomic_torch_save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        save_checkpoint(ckpt_path, args.model_name, model, optimizer, scheduler, epoch, 0, best_f1, best_epoch, patience_counter)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "model": args.model_name,
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_f1_micro": round(train_metrics["f1_micro"], 4),
                "val_loss": round(val_loss, 6),
                "val_f1_micro": round(val_metrics["f1_micro"], 4),
                "val_f1_macro": round(val_metrics["f1_macro"], 4),
                "best_f1": round(best_f1, 4),
                "best_epoch": best_epoch,
            }, ensure_ascii=False) + "\n")
        print(f"Epoch {epoch}/{epochs}: val_F1_micro={val_metrics['f1_micro']:.4f} best={best_f1:.4f} @ {best_epoch}")

        if patience_counter >= patience:
            print(f"Early stopping: no F1 improvement for {patience} epochs.")
            break

    print(f"Finished. Best crop F1 micro={best_f1:.4f} @ epoch {best_epoch}")


if __name__ == "__main__":
    main()
