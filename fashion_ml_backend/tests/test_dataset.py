"""Тести для датасету та transforms."""
import numpy as np
import pytest

from src.dataset import get_transforms, CATEGORY_NAMES, MAIN_APPAREL_IDS


class TestGetTransforms:
    def test_returns_compose(self):
        config = {
            "augmentation": {
                "train": {"resize": [224, 224], "horizontal_flip": 0.5},
                "val": {"resize": [224, 224]},
            }
        }
        t = get_transforms(config, is_train=True)
        assert t is not None

    def test_val_transform(self):
        config = {"augmentation": {"val": {"resize": [224, 224]}}}
        t = get_transforms(config, is_train=False)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        out = t(img)
        assert out.shape == (3, 224, 224)


class TestConstants:
    def test_category_names_count(self):
        assert len(CATEGORY_NAMES) == 46

    def test_main_apparel_ids(self):
        assert len(MAIN_APPAREL_IDS) == 27
        assert 0 in MAIN_APPAREL_IDS
        assert 26 in MAIN_APPAREL_IDS
        assert 27 not in MAIN_APPAREL_IDS
