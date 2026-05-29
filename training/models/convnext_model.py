from __future__ import annotations

import torch.nn as nn
import timm


def build_convnext_model(
    model_name: str,
    num_classes: int,
    pretrained: bool,
    drop_path_rate: float,
) -> nn.Module:
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
