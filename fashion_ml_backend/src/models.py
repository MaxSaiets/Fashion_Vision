"""Model definitions for multi-label classification."""
import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


def get_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """Return a backbone without its classification head."""
    name = name.lower()
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        backbone = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = 2048
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        backbone.classifier = nn.Identity()
        backbone = backbone
        feat_dim = 1280
    elif name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)
        backbone.classifier = nn.Identity()
        feat_dim = 1536
    elif name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        backbone.classifier = nn.Sequential(*(list(backbone.classifier.children())[:2]))
        feat_dim = 768
    elif name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.convnext_small(weights=weights)
        backbone.classifier = nn.Sequential(*(list(backbone.classifier.children())[:2]))
        feat_dim = 768
    else:
        raise ValueError(f"Unknown backbone: {name}")
    return backbone, feat_dim


class MultiLabelClassifier(nn.Module):
    """Multi-label classifier built on top of a vision backbone."""
    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 46,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if self.backbone_name.startswith("resnet"):
            features = features.flatten(1)
        elif self.backbone_name.startswith("convnext"):
            features = features
        else:
            features = features.flatten(1)
        features = self.dropout(features)
        logits = self.fc(features)
        return logits

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
