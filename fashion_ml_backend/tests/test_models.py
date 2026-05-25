"""Тести для моделей."""
import pytest
import torch

from src.models import MultiLabelClassifier, get_backbone


class TestGetBackbone:
    def test_resnet50(self):
        backbone, feat_dim = get_backbone("resnet50", pretrained=False)
        assert feat_dim == 2048
        x = torch.randn(2, 3, 224, 224)
        out = backbone(x)
        assert out.shape == (2, 2048, 1, 1)

    def test_efficientnet_b0(self):
        backbone, feat_dim = get_backbone("efficientnet_b0", pretrained=False)
        assert feat_dim == 1280
        x = torch.randn(2, 3, 224, 224)
        out = backbone(x)
        assert out.shape == (2, 1280)

    def test_unknown_backbone(self):
        with pytest.raises(ValueError, match="Unknown backbone"):
            get_backbone("unknown_model", pretrained=False)


class TestMultiLabelClassifier:
    def test_forward_resnet(self):
        model = MultiLabelClassifier(
            backbone_name="resnet50",
            num_classes=27,
            pretrained=False,
        )
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 27)

    def test_forward_efficientnet(self):
        model = MultiLabelClassifier(
            backbone_name="efficientnet_b0",
            num_classes=46,
            pretrained=False,
        )
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 46)

    def test_freeze_unfreeze(self):
        model = MultiLabelClassifier(
            backbone_name="resnet50",
            num_classes=10,
            pretrained=False,
        )
        model.freeze_backbone()
        for p in model.backbone.parameters():
            assert not p.requires_grad
        model.unfreeze_backbone()
        for p in model.backbone.parameters():
            assert p.requires_grad
