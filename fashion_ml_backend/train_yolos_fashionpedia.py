"""
Fine-tune YOLOS on Fashionpedia detection boxes with resumable checkpoints.
"""
import argparse
import json
import os
import random
import signal
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from src.dataset import CATEGORY_NAMES


STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"\nStop requested by signal {signum}; checkpoint will be saved safely.")


def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_category(category):
    if isinstance(category, str):
        return CATEGORY_NAMES.index(category) if category in CATEGORY_NAMES else None
    if isinstance(category, (list, tuple, np.ndarray)):
        if len(category) == 0:
            return None
        return normalize_category(category[0])
    try:
        value = int(category)
    except (TypeError, ValueError):
        return None
    if 0 <= value < len(CATEGORY_NAMES):
        return value
    return None


def to_coco_bbox(box):
    values = [float(x) for x in box]
    if len(values) != 4:
        return None
    x0, y0, a, b = values
    if a > x0 and b > y0:
        width = a - x0
        height = b - y0
    else:
        width = a
        height = b
    if width <= 1 or height <= 1:
        return None
    return [x0, y0, width, height]


class FashionpediaYolosDataset(IterableDataset):
    def __init__(self, split: str, processor, max_samples: int | None = None, streaming: bool = True):
        self.ds = load_dataset("detection-datasets/fashionpedia", split=split, streaming=streaming)
        self.processor = processor
        self.max_samples = max_samples

    def __iter__(self):
        seen = 0
        for item in self.ds:
            if self.max_samples is not None and seen >= self.max_samples:
                break
            processed = self.process_item(item, seen)
            if processed is not None:
                seen += 1
                yield processed

    def process_item(self, item, idx):
        image = item["image"].convert("RGB") if isinstance(item["image"], Image.Image) else Image.fromarray(np.array(item["image"])).convert("RGB")
        objects = item["objects"]
        annotations = []
        boxes = objects.get("bbox") or objects.get("bboxes") or []
        categories = objects.get("category") or objects.get("categories") or []
        for ann_idx, (box, category) in enumerate(zip(boxes, categories)):
            category_id = normalize_category(category)
            bbox = to_coco_bbox(box)
            if category_id is None or bbox is None:
                continue
            annotations.append(
                {
                    "id": ann_idx,
                    "image_id": int(item.get("image_id", idx)),
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
        target = {"image_id": int(item.get("image_id", idx)), "annotations": annotations}
        encoded = self.processor(images=image, annotations=target, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"][0],
        }


def collate_fn(batch):
    max_h = max(item["pixel_values"].shape[-2] for item in batch)
    max_w = max(item["pixel_values"].shape[-1] for item in batch)
    pixel_values = []
    for item in batch:
        tensor = item["pixel_values"]
        _, h, w = tensor.shape
        padded = torch.zeros((tensor.shape[0], max_h, max_w), dtype=tensor.dtype)
        padded[:, :h, :w] = tensor
        pixel_values.append(padded)
    return {
        "pixel_values": torch.stack(pixel_values),
        "labels": [item["labels"] for item in batch],
    }


def save_checkpoint(path, model, optimizer, scheduler, epoch, batch_idx, best_loss, patience_counter):
    atomic_torch_save(
        {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "best_loss": best_loss,
            "patience_counter": patience_counter,
        },
        path,
    )


def run_epoch(model, loader, optimizer, device, epoch, total_epochs, checkpoint_every, ckpt_path, scheduler, best_loss, patience_counter):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"[yolos] Epoch {epoch}/{total_epochs}", unit="batch", ncols=100)
    for batch_idx, batch in enumerate(pbar, start=1):
        pixel_values = batch["pixel_values"].to(device)
        labels = [{k: v.to(device) for k, v in target.items()} for target in batch["labels"]]
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix(loss=f"{running_loss / batch_idx:.4f}")
        if checkpoint_every and batch_idx % checkpoint_every == 0:
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, batch_idx, best_loss, patience_counter)
        if STOP_REQUESTED:
            break
    return running_loss / max(1, batch_idx)


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="[yolos] Val", unit="batch", leave=False, ncols=80):
            pixel_values = batch["pixel_values"].to(device)
            labels = [{k: v.to(device) for k, v in target.items()} for target in batch["labels"]]
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            batches += 1
    return total_loss / max(1, batches)


def main():
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="valentinafeve/yolos-fashionpedia")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=500)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_streaming", action="store_true", help="Cache the HF dataset locally instead of streaming it.")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    ckpt_path = output_dir / "checkpoint_yolos_fashionpedia.pt"
    best_path = output_dir / "best_yolos_fashionpedia"
    log_path = output_dir / "training_log_yolos_fashionpedia.jsonl"

    from transformers import AutoImageProcessor, YolosForObjectDetection

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    id2label = {idx: name for idx, name in enumerate(CATEGORY_NAMES)}
    label2id = {name: idx for idx, name in id2label.items()}
    model = YolosForObjectDetection.from_pretrained(
        args.model_id,
        num_labels=len(CATEGORY_NAMES),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    streaming = not args.no_streaming
    train_ds = FashionpediaYolosDataset("train", processor, args.max_train_samples, streaming=streaming)
    val_ds = FashionpediaYolosDataset("val", processor, args.max_val_samples, streaming=streaming)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    patience_counter = 0
    start_epoch = 1

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_loss = ckpt.get("best_loss", best_loss)
        patience_counter = ckpt.get("patience_counter", 0)
        saved_epoch = ckpt.get("epoch", 0)
        saved_batch = ckpt.get("batch_idx", 0)
        start_epoch = saved_epoch if saved_batch else saved_epoch + 1
        print(f"Resuming YOLOS from epoch {start_epoch}; best val loss={best_loss:.4f}")
        if saved_batch:
            print(f"Checkpoint was saved mid-epoch at batch {saved_batch}; repeating epoch {saved_epoch} safely.")

    train_count = args.max_train_samples if args.max_train_samples is not None else "stream/all"
    val_count = args.max_val_samples if args.max_val_samples is not None else "stream/all"
    print(f"Device: {device} | train={train_count} val={val_count} labels={len(CATEGORY_NAMES)} streaming={streaming}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, epoch, args.epochs, args.checkpoint_every, ckpt_path, scheduler, best_loss, patience_counter)
        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, 0, best_loss, patience_counter)
        if STOP_REQUESTED:
            print("Stopped after safe checkpoint.")
            return

        val_loss = validate(model, val_loader, device)
        scheduler.step()
        improved = val_loss < best_loss - args.min_delta
        if improved:
            best_loss = val_loss
            patience_counter = 0
            best_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
        else:
            patience_counter += 1

        save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, 0, best_loss, patience_counter)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "model": "yolos_fashionpedia",
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "best_val_loss": round(best_loss, 6),
            }, ensure_ascii=False) + "\n")
        print(f"Epoch {epoch}/{args.epochs}: val_loss={val_loss:.4f} best={best_loss:.4f}")

        if patience_counter >= args.patience:
            print(f"Early stopping: no val loss improvement for {args.patience} epochs.")
            break

    print(f"Finished YOLOS fine-tuning. Best val loss={best_loss:.4f}")


if __name__ == "__main__":
    main()
