"""
Функції втрат для мультилейблової класифікації з підтримкою дисбалансу класів
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss для multi-label: down-weights easy negatives.
    Допомагає при дисбалансі позитивних/негативних зразків.
    """
    def __init__(self, gamma: float = 2.0, reduction: str = "mean", pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) для multi-label - ICCV 2021.
    Різна обробка позитивів/негативів, down-weighting easy negatives.
    Корисно при сильному дисбалансі (багато 0, мало 1).
    """
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_pos = probs
        probs_neg = 1 - probs

        # Clipping для потенційно помилково позначених
        probs_pos = (probs_pos - self.clip).clamp(min=0)
        probs_neg = (probs_neg - self.clip).clamp(min=0)

        # ASL weights
        loss_pos = targets * torch.log(probs_pos + 1e-8)
        loss_neg = (1 - targets) * torch.log(probs_neg + 1e-8)

        loss_pos = loss_pos * (1 - probs_pos) ** self.gamma_pos
        loss_neg = loss_neg * (1 - probs_neg) ** self.gamma_neg

        loss = -(loss_pos + loss_neg)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_fn(
    use_focal: bool = False,
    use_asl: bool = False,
    focal_gamma: float = 2.0,
    asl_gamma_neg: float = 4.0,
    pos_weight: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Повертає функцію втрат для multi-label."""
    if use_asl:
        return AsymmetricLoss(gamma_neg=asl_gamma_neg)
    if use_focal:
        return FocalLoss(gamma=focal_gamma, pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight)


def compute_pos_weight(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Обчислює pos_weight для BCEWithLogitsLoss.
    pos_weight[c] = (кількість негативів) / (кількість позитивів) для класу c
    """
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    pos = labels.sum(dim=0)
    neg = labels.shape[0] - pos
    # Уникаємо ділення на нуль
    weight = neg / (pos + 1e-6)
    weight = torch.clamp(weight, 0.1, 10.0)
    return weight
