# src/model.py
import torch.nn as nn
from torchvision.models import mobilenet_v2

def build_model(num_classes=10, width_mult=1.4, dropout=0.2):
    model = mobilenet_v2(width_mult=width_mult)

    # -----------------------------
    # IMPORTANT: CIFAR-10 FIX
    # Change first conv stride from 2 → 1
    # -----------------------------
    model.features[0][0].stride = (1, 1)

    # Replace classifier for CIFAR-10
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_f, num_classes)
    )

    return model
