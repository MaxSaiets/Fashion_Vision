"""Training script for multi-label clothing attribute classification."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.dataset import FashionpediaHFDataset, get_transforms, CATEGORY_NAMES
from src.models import MultiLabelClassifier
from src.losses import get_loss_fn, compute_pos_weight
from src.metrics import multi_label_metrics, per_class_f1
from src.charts import (
    plot_training_curves,
    plot_models_comparison,
    plot_class_distribution,
    plot_per_class_f1,
    plot_confusion_heatmap_multilabel,
    generate_summary_report,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    """Форматує час у читабельний вигляд."""
    if seconds < 60:
        return f"{seconds:.1f}с"
    elif seconds < 3600:
        return f"{seconds/60:.1f}хв"
    return f"{seconds/3600:.1f}год"


def train_epoch(model, loader, optimizer, loss_fn, device, epoch: int, total_epochs: int, model_name: str):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    num_batches = len(loader)
    start_time = time.time()

    pbar = tqdm(
        loader,
        desc=f"[{model_name}] Epoch {epoch}/{total_epochs} Train",
        unit="batch",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for batch_idx, (batch_x, batch_y) in enumerate(pbar):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch_y.cpu().numpy())

        # Оновлення progress bar
        avg_loss = total_loss / (batch_idx + 1)
        elapsed = time.time() - start_time
        rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
        eta = (num_batches - batch_idx - 1) / rate if rate > 0 else 0
        pbar.set_postfix(loss=f"{avg_loss:.4f}", ETA=f"{format_time(eta)}")

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = multi_label_metrics(all_labels, all_preds)
    return total_loss / len(loader), metrics


def validate(model, loader, loss_fn, device, desc: str = "Val"):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(loader, desc=desc, unit="batch", leave=False, ncols=80):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            total_loss += loss.item()
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(batch_y.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = multi_label_metrics(all_labels, all_preds)
    return total_loss / len(loader), metrics, all_preds, all_labels


def should_stop_early(
    val_metrics: dict,
    best_metrics: dict,
    patience_counter: int,
    patience: int,
    min_delta: float,
    plateau_window: int,
    recent_f1: list,
) -> tuple:
    """
    Розумне early stopping:
    - Немає покращення F1 за patience епох
    - Loss plateau (не зменшується)
    - F1 почав падати стійко
    """
    stop = False
    reason = ""

    if patience_counter >= patience:
        stop = True
        reason = f"Немає покращення F1 за {patience} епох (min_delta={min_delta})"

    # Plateau: останні N епох F1 майже не змінюється
    if plateau_window >= 3 and len(recent_f1) >= plateau_window:
        std = np.std(recent_f1[-plateau_window:])
        if std < 0.005 and not stop:
            stop = True
            reason = f"Plateau: F1 стабілізувався (std={std:.4f})"

    return stop, reason


def save_checkpoint(
    output_dir: Path,
    model_name: str,
    model,
    optimizer,
    scheduler,
    epoch: int,
    best_f1: float,
    best_epoch: int,
    patience_counter: int,
    recent_f1: list,
):
    """Зберігає checkpoint для продовження навчання."""
    ckpt = {
        "model_name": model_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_f1": best_f1,
        "best_epoch": best_epoch,
        "patience_counter": patience_counter,
        "recent_f1": recent_f1,
    }
    torch.save(ckpt, output_dir / f"checkpoint_{model_name}.pt")
    last_run = {"model": model_name, "epoch": epoch}
    with open(output_dir / "last_run.json", "w", encoding="utf-8") as f:
        json.dump(last_run, f, indent=2)


def load_resume_state(output_dir: Path) -> dict | None:
    """Повертає інфо для resume або None якщо немає checkpoint."""
    last_path = output_dir / "last_run.json"
    if last_path.exists():
        with open(last_path, "r", encoding="utf-8") as f:
            return json.load(f)
    log_path = output_dir / "training_log.jsonl"
    if not log_path.exists():
        return None
    last_model, last_epoch = None, 0
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                last_model = entry.get("model")
                last_epoch = entry.get("epoch", 0)
            except json.JSONDecodeError:
                continue
    if last_model and last_epoch > 0:
        return {"model": last_model, "epoch": last_epoch}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Шлях до config.yaml")
    parser.add_argument("--max_samples", type=int, default=None, help="Ліміт зразків для тесту")
    parser.add_argument("--resume", action="store_true", help="Продовжити з останнього checkpoint")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.max_samples:
        config["dataset"]["max_samples"] = args.max_samples

    set_seed(config.get("project", {}).get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.get("project", {}).get("output_dir", "outputs"))
    charts_dir = Path(config.get("metrics", {}).get("charts_dir", "outputs/charts"))
    log_path = output_dir / "training_log.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    resume_info = load_resume_state(output_dir) if args.resume else None
    if not args.resume and log_path.exists():
        log_path.write_text("")

    print("\n" + "=" * 70)
    print("  МУЛЬТИЛЕЙБЛОВА КЛАСИФІКАЦІЯ АТРИБУТІВ ОДЯГУ - Fashionpedia")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    if resume_info:
        print(f"  RESUME: {resume_info['model']} з епохи {resume_info['epoch']}")
    print("=" * 70 + "\n")

    # Dataset
    ds_config = config["dataset"]
    mode = ds_config.get("mode", "huggingface")
    main_apparel = ds_config.get("main_apparel_only", True)
    max_samples = ds_config.get("max_samples")

    transform_train = get_transforms(config, is_train=True)
    transform_val = get_transforms(config, is_train=False)

    if mode == "huggingface":
        print("Завантаження датасету (HuggingFace)...")
        train_ds = FashionpediaHFDataset(
            split="train",
            transform=transform_train,
            main_apparel_only=main_apparel,
            max_samples=max_samples,
        )
        val_ds = FashionpediaHFDataset(
            split="val",
            transform=transform_val,
            main_apparel_only=main_apparel,
            max_samples=max_samples,
        )
        num_classes = train_ds.num_classes
        class_names = CATEGORY_NAMES[:num_classes]
    elif mode == "fashionpedia_json":
        from src.dataset import FashionpediaJSONDataset
        ann_dir = Path(ds_config.get("annotations_dir", "data/annotations"))
        img_dir = Path(ds_config.get("images_dir", "data/images"))
        train_ds = FashionpediaJSONDataset(
            split="train",
            annotations_path=str(ann_dir / "attributes_train2020.json"),
            images_dir=str(img_dir),
            transform=transform_train,
            max_samples=max_samples,
        )
        val_ds = FashionpediaJSONDataset(
            split="val",
            annotations_path=str(ann_dir / "attributes_val2020.json"),
            images_dir=str(img_dir),
            transform=transform_val,
            max_samples=max_samples,
        )
        num_classes = train_ds.num_attributes
        class_names = getattr(train_ds, "class_names", None)
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")

    print(f"  Train: {len(train_ds)} зразків | Val: {len(val_ds)} | Класів: {num_classes}\n")

    # Class distribution chart
    all_labels = np.array([train_ds.samples[i]["labels"] for i in range(len(train_ds))])
    plot_class_distribution(
        all_labels,
        class_names=class_names,
        output_path=str(charts_dir / "class_distribution.png"),
        top_k=min(30, num_classes),
    )

    # Pos weight for imbalance
    all_labels_t = torch.from_numpy(all_labels)
    pos_weight = compute_pos_weight(all_labels_t, num_classes).to(device) if config["training"].get("use_pos_weight") else None

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    train_cfg = config["training"]
    models_cfg = config["models"]
    patience = train_cfg.get("patience", 5)
    min_delta = train_cfg.get("min_delta", 0.001)
    plateau_window = train_cfg.get("plateau_window", 4)

    for model_idx, model_cfg in enumerate(models_cfg):
        model_name = model_cfg["name"]
        model_start = time.time()

        # Пропускаємо моделі до resume
        resume_model = resume_info["model"] if resume_info else None
        resume_epoch = resume_info.get("epoch", 0) if resume_info else 0
        if resume_info and resume_info["model"] != model_name:
            print(f"\n  Пропуск {model_name} (resume з {resume_info['model']})")
            continue
        if resume_info and resume_info["model"] == model_name:
            resume_info = None  # використали, далі не resume

        print("\n" + "-" * 70)
        print(f"  МОДЕЛЬ: {model_name} ({model_cfg['backbone']})")
        print("-" * 70)

        model = MultiLabelClassifier(
            backbone_name=model_cfg["backbone"],
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", True),
        ).to(device)

        freeze_epochs = model_cfg.get("freeze_backbone_epochs", 0)
        if freeze_epochs > 0:
            model.freeze_backbone()
            print(f"  Freeze backbone: перші {freeze_epochs} епох")

        loss_fn = get_loss_fn(
            use_focal=train_cfg.get("use_focal_loss", False),
            focal_gamma=train_cfg.get("focal_gamma", 2.0),
            pos_weight=pos_weight,
        )

        lr = float(train_cfg.get("learning_rate", 1e-4))
        wd = float(train_cfg.get("weight_decay", 1e-4))
        min_lr = float(train_cfg.get("min_lr", 1e-6))

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )

        scheduler = None
        if train_cfg.get("scheduler") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_cfg["num_epochs"],
                eta_min=min_lr,
            )

        best_f1 = 0.0
        best_epoch = 0
        patience_counter = 0
        recent_f1 = []
        start_epoch = 1

        # Завантаження checkpoint при --resume
        ckpt_path = output_dir / f"checkpoint_{model_name}.pt"
        best_path = output_dir / f"best_{model_name}.pt"
        if args.resume and ckpt_path.exists():
            print(f"  Завантаження checkpoint...")
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") and scheduler:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_f1 = ckpt["best_f1"]
            best_epoch = ckpt["best_epoch"]
            patience_counter = ckpt["patience_counter"]
            recent_f1 = ckpt.get("recent_f1", [])
            print(f"  Продовження з епохи {start_epoch} (best F1={best_f1:.4f} @ epoch {best_epoch})")
            if start_epoch > train_cfg["num_epochs"]:
                print(f"  Модель вже навчена до кінця, пропуск.")
                continue
        elif args.resume and best_path.exists() and resume_model == model_name:
            # Fallback: немає checkpoint, але є best weights (старий запуск)
            print(f"  Checkpoint відсутній, завантаження best weights...")
            model.load_state_dict(torch.load(best_path, map_location=device))
            start_epoch = resume_epoch + 1
            print(f"  Продовження з епохи {start_epoch} (optimizer скинуто)")
            if start_epoch > train_cfg["num_epochs"]:
                print(f"  Модель вже навчена до кінця, пропуск.")
                continue

        for epoch in range(start_epoch, train_cfg["num_epochs"] + 1):
            epoch_start = time.time()

            if epoch == freeze_epochs + 1 and freeze_epochs > 0:
                model.unfreeze_backbone()
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=lr,
                    weight_decay=wd,
                )
                if scheduler:
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=train_cfg["num_epochs"] - freeze_epochs,
                        eta_min=min_lr,
                    )
                print(f"  Unfreeze backbone на епосі {epoch}")

            train_loss, train_metrics = train_epoch(
                model, train_loader, optimizer, loss_fn, device,
                epoch, train_cfg["num_epochs"], model_name,
            )
            val_loss, val_metrics, val_preds, val_labels = validate(
                model, val_loader, loss_fn, device,
                desc=f"[{model_name}] Epoch {epoch} Val",
            )

            if scheduler:
                scheduler.step()

            recent_f1.append(val_metrics["f1_micro"])
            if len(recent_f1) > plateau_window:
                recent_f1.pop(0)

            log_entry = {
                "model": model_name,
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_f1_micro": round(train_metrics["f1_micro"], 4),
                "val_loss": round(val_loss, 6),
                "val_f1_micro": round(val_metrics["f1_micro"], 4),
                "val_f1_macro": round(val_metrics["f1_macro"], 4),
                "val_hamming_score": round(val_metrics["hamming_score"], 4),
            }

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            epoch_time = time.time() - epoch_start
            eta_total = epoch_time * (train_cfg["num_epochs"] - epoch) if epoch < train_cfg["num_epochs"] else 0

            # Вивід в консоль
            improved = " *" if val_metrics["f1_micro"] > best_f1 else ""
            print(f"\n  Epoch {epoch:2d}/{train_cfg['num_epochs']} | "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
                  f"val_F1_micro={val_metrics['f1_micro']:.4f} val_F1_macro={val_metrics['f1_macro']:.4f}{improved}")
            print(f"  Час епохи: {format_time(epoch_time)} | Залишилось ~{format_time(eta_total)}")

            if val_metrics["f1_micro"] > best_f1:
                best_f1 = val_metrics["f1_micro"]
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / f"best_{model_name}.pt")
            else:
                patience_counter += 1

            # Checkpoint для resume (після кожної епохи)
            save_checkpoint(
                output_dir, model_name, model, optimizer, scheduler,
                epoch, best_f1, best_epoch, patience_counter, recent_f1,
            )

            stop, reason = should_stop_early(
                val_metrics, {"f1_micro": best_f1},
                patience_counter, patience, min_delta, plateau_window, recent_f1,
            )
            if stop:
                print(f"\n  >>> EARLY STOPPING: {reason}")
                break

        model_time = time.time() - model_start
        print(f"\n  [{model_name}] Навчання завершено за {format_time(model_time)}")
        print(f"  Best F1 micro: {best_f1:.4f} (epoch {best_epoch})")

        # Charts per model
        plot_training_curves(str(log_path), str(charts_dir), model_name=model_name)

        # Per-class F1 for best model
        model.load_state_dict(torch.load(output_dir / f"best_{model_name}.pt", map_location=device))
        _, _, val_preds, val_labels = validate(model, val_loader, loss_fn, device, desc="Final eval")
        f1_per_cls = per_class_f1(val_labels, val_preds)
        plot_per_class_f1(
            f1_per_cls,
            class_names=class_names,
            output_path=str(charts_dir / f"per_class_f1_{model_name}.png"),
            top_k=min(30, num_classes),
        )
        plot_confusion_heatmap_multilabel(
            val_labels,
            val_preds,
            class_names=class_names,
            output_path=str(charts_dir / f"label_correlation_{model_name}.png"),
            top_k=min(20, num_classes),
        )

    # Порівняння моделей та звіт
    plot_models_comparison(str(log_path), str(charts_dir))
    generate_summary_report(str(log_path), str(charts_dir))

    print("\n" + "=" * 70)
    print("  НАВЧАННЯ ЗАВЕРШЕНО")
    print("=" * 70)
    print(f"  Результати: {output_dir}")
    print(f"  Графіки:    {charts_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
