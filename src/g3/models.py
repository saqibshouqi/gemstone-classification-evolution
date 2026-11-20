import torch.nn as nn
from torchvision import models

from src.g3.cbam import CBAM


def build_g3_model(model_name: str, num_classes: int = 68, pretrained: bool = True):
    """Build a G3 backbone with a new 68-class head and frozen backbone (Stage A)."""
    if model_name == "densenet169":
        weights = models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet169(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        head_params = list(model.classifier.parameters())

    elif model_name == "convnext_small":
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_small(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        for p in model.classifier.parameters():
            p.requires_grad = True
        head_params = list(model.classifier.parameters())

    elif model_name == "convnext_small_cbam":
        weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.convnext_small(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        model.features = nn.Sequential(*list(model.features.children()), CBAM(channels=768))
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        for p in model.classifier.parameters():
            p.requires_grad = True
        head_params = list(model.features[8].parameters()) + list(model.classifier.parameters())

    elif model_name == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        head_params = list(model.classifier.parameters())

    else:
        raise ValueError(f"Unknown G3 model_name: {model_name}")

    return model, head_params


def unfreeze_deeper_layers(model: nn.Module, model_name: str):
    """Unfreeze the final dense block / stage for Stage B fine-tuning."""
    backbone_params = []

    if model_name == "densenet169":
        for layer in [model.features.denseblock4, model.features.norm5]:
            for p in layer.parameters():
                p.requires_grad = True
                backbone_params.append(p)
        head_params = list(model.classifier.parameters())

    elif model_name == "convnext_small":
        for idx in [6, 7]:
            for p in model.features[idx].parameters():
                p.requires_grad = True
                backbone_params.append(p)
        head_params = list(model.classifier.parameters())

    elif model_name == "convnext_small_cbam":
        for idx in [6, 7]:
            for p in model.features[idx].parameters():
                p.requires_grad = True
                backbone_params.append(p)
        head_params = list(model.features[8].parameters()) + list(model.classifier.parameters())

    elif model_name == "mobilenetv2":
        for idx in range(14, len(model.features)):
            for p in model.features[idx].parameters():
                p.requires_grad = True
                backbone_params.append(p)
        head_params = list(model.classifier.parameters())

    else:
        raise ValueError(f"Unknown G3 model_name: {model_name}")

    return backbone_params, head_params