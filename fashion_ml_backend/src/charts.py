"""Plotting helpers for model training and evaluation."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_training_log(log_path: str) -> List[Dict]:
    """Load a JSONL training log."""
    records = []
    path = Path(log_path)
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def plot_training_curves(
    log_path: str,
    output_dir: str,
    model_name: str = "model",
) -> None:
    """Plot loss and metric curves across epochs."""
    records = load_training_log(log_path)
    if not records:
        return

    model_records = [r for r in records if r.get("model") == model_name]
    if not model_records:
        model_records = records

    epochs = [r["epoch"] for r in model_records if "epoch" in r]
    train_loss = [r["train_loss"] for r in model_records if "train_loss" in r]
    val_loss = [r.get("val_loss") for r in model_records if "val_loss" in r]
    val_f1 = [r.get("val_f1_micro") for r in model_records if "val_f1_micro" in r]
    train_f1 = [r.get("train_f1_micro") for r in model_records if "train_f1_micro" in r]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax1 = axes[0, 0]
    ax1.plot(epochs, train_loss, "b-o", label="Train Loss", markersize=4)
    if any(v is not None for v in val_loss):
        ax1.plot(epochs, val_loss, "r-s", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss - {model_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if val_f1 and any(v is not None for v in val_f1):
        ax2.plot(epochs, val_f1, "g-^", label="Val F1 Micro", markersize=4)
    if train_f1 and any(v is not None for v in train_f1):
        ax2.plot(epochs, train_f1, "b--o", label="Train F1 Micro", markersize=3, alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Micro")
    ax2.set_title(f"F1 Score - {model_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    val_f1_macro = [r.get("val_f1_macro") for r in model_records if "val_f1_macro" in r]
    if val_f1_macro and any(v is not None for v in val_f1_macro):
        ax3.plot(epochs, val_f1_macro, "m-^", label="Val F1 Macro", markersize=4)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("F1 Macro")
    ax3.set_title(f"F1 Macro - {model_name}")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    hamming = [r.get("val_hamming_score") for r in model_records if "val_hamming_score" in r]
    if hamming and any(v is not None for v in hamming):
        ax4.plot(epochs, hamming, "c-^", label="Val Hamming Score", markersize=4)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Hamming Score")
    ax4.set_title(f"Hamming Score - {model_name}")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        Path(output_dir) / f"training_curves_{model_name}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plot_models_comparison(
    log_path: str,
    output_dir: str,
    metrics: List[str] = ["val_f1_micro", "val_loss", "val_f1_macro"],
) -> None:
    """Plot metric comparisons for multiple models."""
    records = load_training_log(log_path)
    if not records:
        return

    models = list(set(r.get("model", "default") for r in records if "model" in r))
    if len(models) < 2:
        models = ["model"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for model in models:
            model_records = [r for r in records if r.get("model") == model]
            epochs = [r["epoch"] for r in model_records if "epoch" in r and metric in r]
            values = [r[metric] for r in model_records if metric in r]
            if epochs and values:
                ax.plot(epochs, values, "-o", label=model, markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.set_title(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        Path(output_dir) / "models_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "class_distribution.png",
    top_k: int = 30,
) -> None:
    """Plot class-frequency distribution."""
    counts = labels.sum(axis=0)
    indices = np.argsort(counts)[::-1][:top_k]
    counts = counts[indices]

    if class_names:
        names = [class_names[i] if i < len(class_names) else str(i) for i in indices]
    else:
        names = [str(i) for i in indices]

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(names)), counts, color="steelblue", alpha=0.8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xlabel("Кількість зразків")
    plt.title("Розподіл класів (топ-%d)" % top_k)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_class_f1(
    f1_per_class: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "per_class_f1.png",
    top_k: int = 30,
) -> None:
    """Plot per-class F1 scores."""
    indices = np.argsort(f1_per_class)[::-1][:top_k]
    values = f1_per_class[indices]

    if class_names:
        names = [class_names[i] if i < len(class_names) else str(i) for i in indices]
    else:
        names = [str(i) for i in indices]

    plt.figure(figsize=(12, 6))
    colors = ["green" if v > 0.5 else "orange" if v > 0.2 else "red" for v in values]
    plt.barh(range(len(names)), values, color=colors, alpha=0.8)
    plt.yticks(range(len(names)), names, fontsize=8)
    plt.xlabel("F1 Score")
    plt.title("F1 по класах (топ-%d)" % top_k)
    plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_heatmap_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "label_correlation.png",
    top_k: int = 20,
) -> None:
    """Plot precision and recall by class for multi-label predictions."""
    num_classes = y_true.shape[1]
    y_pred_bin = (y_pred >= 0.5).astype(np.float32)

    tp = (y_true * y_pred_bin).sum(axis=0)
    fp = ((1 - y_true) * y_pred_bin).sum(axis=0)
    fn = (y_true * (1 - y_pred_bin)).sum(axis=0)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    support = y_true.sum(axis=0)
    top_indices = np.argsort(support)[::-1][:top_k]

    data = np.stack([precision[top_indices], recall[top_indices]], axis=1)
    names = [class_names[i] if class_names and i < len(class_names) else str(i) for i in top_indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data.T, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Precision", "Recall"])
    plt.colorbar(im, ax=ax, label="Score")
    plt.title("Precision & Recall по класах")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_summary_report(log_path: str, output_dir: str) -> None:
    """Write a compact summary report for the recorded training runs."""
    records = load_training_log(log_path)
    if not records:
        return

    models = {}
    for r in records:
        name = r.get("model", "model")
        if name not in models:
            models[name] = []
        models[name].append(r)

    rows = []
    for model_name, recs in models.items():
        if not recs:
            continue
        best = max(recs, key=lambda x: x.get("val_f1_micro", 0))
        last = recs[-1]
        rows.append({
            "model": model_name,
            "best_f1_micro": best.get("val_f1_micro", 0),
            "best_f1_macro": best.get("val_f1_macro", 0),
            "best_epoch": best.get("epoch", 0),
            "final_epoch": last.get("epoch", 0),
            "final_val_loss": last.get("val_loss", 0),
        })

    csv_path = Path(output_dir) / "summary_results.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    txt_path = Path(output_dir) / "summary_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("TRAINING SUMMARY - Fashionpedia Multi-Label Classification\n")
        f.write("=" * 60 + "\n\n")
        for r in rows:
            f.write(f"Model: {r['model']}\n")
            f.write(f"  Best F1 micro: {r['best_f1_micro']:.4f} (epoch {r['best_epoch']})\n")
            f.write(f"  Best F1 macro: {r['best_f1_macro']:.4f}\n")
            f.write(f"  Epochs trained: {r['final_epoch']}\n\n")
