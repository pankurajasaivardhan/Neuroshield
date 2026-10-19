# Track 1 Model B: ViT-B/16 (GPU: Kaggle/Colab).
from __future__ import annotations
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from .train_model_a import run

def build_model() -> nn.Module:
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.encoder.layers[-1].parameters():
        p.requires_grad = True
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    return model
if __name__ == '__main__':
    run(model_builder=build_model, model_name='model_b_vit')
