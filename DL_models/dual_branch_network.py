# -*- coding: utf-8 -*-
"""
Upgraded Dual-Branch Multimodal Architecture for Student Engagement Recognition.
Combines:
1. Deep Visual Representation: EfficientNet-B0 (1,280 dims) / ResNet-18 (512 dims)
2. Fine-grained Action Geometry: 52 MediaPipe FACS Blendshapes (64 dims)
3. Regularized Multimodal Fusion Head with LayerNorm & GELU
"""

import torch
import torch.nn as nn
from torchvision import models

class FACSBlendshapesBranch(nn.Module):
    """
    Processes 52 MediaPipe Facial Action Coding System (FACS) blendshape scores.
    Projects muscle activations into a compact, regularized 64-dimensional feature space.
    """
    def __init__(self, in_features=52, out_features=64, dropout_rate=0.10):
        super(FACSBlendshapesBranch, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, out_features),
            nn.LayerNorm(out_features),
            nn.GELU()
        )

    def forward(self, x):
        return self.fc(x)


class StudentEngagementDualBranchModel(nn.Module):
    """
    Dual-Branch Multimodal Fusion Network:
    - Visual Branch: Deep CNN backbone (EfficientNet-B0 or ResNet-18)
    - Geometry Branch: FACSBlendshapesBranch (52 blendshape metrics)
    - Multimodal Fusion Head: LayerNorm -> Linear(256) -> GELU -> Dropout -> Linear(num_classes)
    """
    def __init__(self, backbone_type="efficientnet", num_classes=3,
                 blendshape_dim=52, fusion_hidden_dim=256, dropout_rate=0.25,
                 pretrained=True, image_only=False):
        super(StudentEngagementDualBranchModel, self).__init__()
        self.backbone_type = backbone_type.lower()
        self.image_only = image_only

        # 1. Visual Feature Extractor
        if self.backbone_type == "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            eff = models.efficientnet_b1(weights=weights)
            self.visual_backbone = eff.features
            self.avgpool = eff.avgpool
            self.visual_dim = eff.classifier[1].in_features  # 1280
        elif self.backbone_type in ["efficientnet", "efficientnet_b0"]:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            eff = models.efficientnet_b0(weights=weights)
            self.visual_backbone = eff.features
            self.avgpool = eff.avgpool
            self.visual_dim = eff.classifier[1].in_features  # 1280
        elif self.backbone_type == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            res = models.resnet34(weights=weights)
            self.visual_backbone = nn.Sequential(
                res.conv1, res.bn1, res.relu, res.maxpool,
                res.layer1, res.layer2, res.layer3, res.layer4
            )
            self.avgpool = res.avgpool
            self.visual_dim = res.fc.in_features  # 512
        elif self.backbone_type == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            res = models.resnet50(weights=weights)
            self.visual_backbone = nn.Sequential(
                res.conv1, res.bn1, res.relu, res.maxpool,
                res.layer1, res.layer2, res.layer3, res.layer4
            )
            self.avgpool = res.avgpool
            self.visual_dim = res.fc.in_features  # 2048
        elif self.backbone_type in ["resnet", "resnet18"]:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            res = models.resnet18(weights=weights)
            self.visual_backbone = nn.Sequential(
                res.conv1, res.bn1, res.relu, res.maxpool,
                res.layer1, res.layer2, res.layer3, res.layer4
            )
            self.avgpool = res.avgpool
            self.visual_dim = res.fc.in_features  # 512
        elif self.backbone_type in ["mobilenet_v3", "mobilenet_v3_large", "mobilenet"]:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            mob = models.mobilenet_v3_large(weights=weights)
            self.visual_backbone = mob.features
            self.avgpool = mob.avgpool
            self.visual_dim = mob.classifier[0].in_features  # 960
        elif self.backbone_type == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            mob = models.mobilenet_v3_small(weights=weights)
            self.visual_backbone = mob.features
            self.avgpool = mob.avgpool
            self.visual_dim = mob.classifier[0].in_features  # 576
        else:
            raise ValueError(f"Unsupported backbone_type: {backbone_type}. Supported: resnet18, resnet34, resnet50, efficientnet_b0, efficientnet_b1, mobilenet_v3, mobilenet_v3_small")

        # 2. FACS Blendshapes Geometric Branch (Skipped in Image-Only mode)
        if not self.image_only:
            self.blendshape_projector = FACSBlendshapesBranch(
                in_features=blendshape_dim,
                out_features=64,
                dropout_rate=0.10
            )
            self.blendshape_dim_out = 64
        else:
            self.blendshape_projector = None
            self.blendshape_dim_out = 0

        # 3. Classification Head
        total_features = self.visual_dim + self.blendshape_dim_out
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(total_features),
            nn.Linear(total_features, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def extract_visual_features(self, images):
        x = self.visual_backbone(images)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, images, blendshapes=None):
        # Branch A: Deep visual features
        visual_feat = self.extract_visual_features(images)

        # Image-only ablation mode (like Riju IF)
        if self.image_only or blendshapes is None:
            return self.fusion_head(visual_feat)

        # Branch B: FACS Blendshape scores
        geom_feat = self.blendshape_projector(blendshapes)

        # Multimodal Concatenation
        fused = torch.cat([visual_feat, geom_feat], dim=1)

        # Classification Logits
        logits = self.fusion_head(fused)
        return logits


def get_upgraded_model(backbone_type="efficientnet", num_classes=3, dropout_rate=0.25, image_only=False):
    """
    Factory function to construct the Engagement model (Dual-Branch or Image-Only).
    """
    return StudentEngagementDualBranchModel(
        backbone_type=backbone_type,
        num_classes=num_classes,
        blendshape_dim=52,
        fusion_hidden_dim=256,
        dropout_rate=dropout_rate,
        pretrained=True,
        image_only=image_only
    )
