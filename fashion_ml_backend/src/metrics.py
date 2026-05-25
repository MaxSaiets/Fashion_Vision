"""
Метрики для мультилейблової класифікації
"""
import numpy as np
import torch
from typing import Dict, List, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    roc_auc_score,
)


def multi_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Обчислює метрики для multi-label класифікації.
    y_true, y_pred: (N, num_classes) - бінарні або ймовірності
    """
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    
    # Subset accuracy (exact match) - рідко використовується
    exact_match = np.mean(np.all(y_true == y_pred_bin, axis=1))
    
    # Hamming score = 1 - hamming_loss
    hamming = hamming_loss(y_true, y_pred_bin)
    hamming_score = 1 - hamming
    
    # F1: micro (глобальний), macro (середнє по класах), samples (середнє по зразках)
    f1_micro = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    f1_samples = f1_score(y_true, y_pred_bin, average="samples", zero_division=0)
    
    precision_micro = precision_score(y_true, y_pred_bin, average="micro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred_bin, average="micro", zero_division=0)
    
    # ROC-AUC (використовує ймовірності, не бінарні мітки)
    try:
        roc_auc_micro = roc_auc_score(y_true, y_pred, average="micro")
    except ValueError:
        roc_auc_micro = 0.0
    try:
        roc_auc_macro = roc_auc_score(y_true, y_pred, average="macro")
    except ValueError:
        roc_auc_macro = 0.0
    
    return {
        "exact_match": float(exact_match),
        "hamming_score": float(hamming_score),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_samples": float(f1_samples),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "roc_auc_micro": float(roc_auc_micro),
        "roc_auc_macro": float(roc_auc_macro),
    }


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """F1 для кожного класу окремо."""
    y_pred_bin = (y_pred >= threshold).astype(np.float32)
    num_classes = y_true.shape[1]
    f1_per_class = np.zeros(num_classes)
    for c in range(num_classes):
        f1_per_class[c] = f1_score(
            y_true[:, c], y_pred_bin[:, c], zero_division=0
        )
    return f1_per_class
