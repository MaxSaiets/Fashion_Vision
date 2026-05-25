"""Pytest fixtures."""
import numpy as np
import torch
import pytest


@pytest.fixture
def sample_labels():
    """Multi-label targets: 5 samples, 10 classes."""
    return torch.tensor([
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    ], dtype=torch.float32)


@pytest.fixture
def sample_logits():
    """Random logits for testing."""
    return torch.randn(5, 10)


@pytest.fixture
def sample_y_true():
    """Numpy labels for metrics."""
    return np.array([
        [1, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 1],
    ], dtype=np.float32)


@pytest.fixture
def sample_y_pred():
    """Numpy predictions (probabilities)."""
    return np.array([
        [0.9, 0.1, 0.8, 0.2],
        [0.2, 0.9, 0.7, 0.1],
        [0.8, 0.7, 0.3, 0.9],
    ], dtype=np.float32)
