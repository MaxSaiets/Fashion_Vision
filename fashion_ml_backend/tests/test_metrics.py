"""Тести для метрик."""
import pytest
import numpy as np

from src.metrics import multi_label_metrics, per_class_f1


class TestMultiLabelMetrics:
    def test_returns_dict(self, sample_y_true, sample_y_pred):
        m = multi_label_metrics(sample_y_true, sample_y_pred)
        assert isinstance(m, dict)
        assert "f1_micro" in m
        assert "f1_macro" in m
        assert "hamming_score" in m
        assert "exact_match" in m

    def test_values_in_range(self, sample_y_true, sample_y_pred):
        m = multi_label_metrics(sample_y_true, sample_y_pred)
        for k, v in m.items():
            assert 0 <= v <= 1 or (k == "exact_match" and 0 <= v <= 1)

    def test_perfect_prediction(self):
        y = np.array([[1, 0], [0, 1]], dtype=np.float32)
        m = multi_label_metrics(y, y)
        assert m["exact_match"] == 1.0
        assert m["f1_micro"] == 1.0

    def test_custom_threshold(self, sample_y_true, sample_y_pred):
        m = multi_label_metrics(sample_y_true, sample_y_pred, threshold=0.6)
        assert "f1_micro" in m


class TestPerClassF1:
    def test_shape(self, sample_y_true, sample_y_pred):
        f1 = per_class_f1(sample_y_true, sample_y_pred)
        assert f1.shape == (sample_y_true.shape[1],)

    def test_values_in_range(self, sample_y_true, sample_y_pred):
        f1 = per_class_f1(sample_y_true, sample_y_pred)
        assert (f1 >= 0).all()
        assert (f1 <= 1).all()
