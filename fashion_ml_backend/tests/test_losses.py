"""Тести для функцій втрат."""
import pytest
import torch

from src.losses import (
    FocalLoss,
    AsymmetricLoss,
    get_loss_fn,
    compute_pos_weight,
)


class TestFocalLoss:
    def test_forward_shape(self, sample_logits, sample_labels):
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(sample_logits, sample_labels)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_reduction_sum(self, sample_logits, sample_labels):
        loss_fn = FocalLoss(gamma=2.0, reduction="sum")
        loss = loss_fn(sample_logits, sample_labels)
        assert loss.dim() == 0
        assert loss.item() > 0


class TestAsymmetricLoss:
    def test_forward_shape(self, sample_logits, sample_labels):
        loss_fn = AsymmetricLoss(gamma_neg=4.0)
        loss = loss_fn(sample_logits, sample_labels)
        assert loss.dim() == 0
        assert loss.item() > 0


class TestGetLossFn:
    def test_bce(self):
        fn = get_loss_fn()
        assert fn.__class__.__name__ == "BCEWithLogitsLoss"

    def test_focal(self):
        fn = get_loss_fn(use_focal=True)
        assert isinstance(fn, FocalLoss)

    def test_asl(self):
        fn = get_loss_fn(use_asl=True)
        assert isinstance(fn, AsymmetricLoss)


class TestComputePosWeight:
    def test_shape(self, sample_labels):
        w = compute_pos_weight(sample_labels, num_classes=10)
        assert w.shape == (10,)

    def test_values_bounded(self, sample_labels):
        w = compute_pos_weight(sample_labels, num_classes=10)
        assert (w >= 0.1).all()
        assert (w <= 10.0).all()

    def test_single_sample(self):
        labels = torch.tensor([[1, 0, 1]], dtype=torch.float32)
        w = compute_pos_weight(labels, num_classes=3)
        assert w.shape == (3,)
