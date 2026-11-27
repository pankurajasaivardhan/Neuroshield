# Uncertainty: predictive entropy, mutual information, MC Dropout.
from __future__ import annotations
import numpy as np
try:
    import torch
except ImportError:
    torch = None

def predictive_entropy(mean_probs: np.ndarray, eps: float=1e-12) -> np.ndarray:
    p = np.clip(mean_probs, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)

def mutual_information(sample_probs: np.ndarray, eps: float=1e-12) -> np.ndarray:
    mean_p = sample_probs.mean(axis=0)
    total = predictive_entropy(mean_p, eps)
    per_sample = -np.sum(np.clip(sample_probs, eps, 1.0) * np.log(np.clip(sample_probs, eps, 1.0)), axis=-1)
    return total - per_sample.mean(axis=0)

def _enable_dropout(model) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()
            if m.p > 0:
                n += 1
    return n


def mc_dropout_probs(model, inputs, n_passes: int=30) -> np.ndarray:
    """MC Dropout. Returns (n_passes, batch, classes).

    Models whose Dropout modules are missing (torchvision ResNet-50) or have p=0 (torchvision
    ViT-B/16 with its default dropout=0.0 and attention_dropout=0.0) give identical passes,
    so mutual information is exactly 0. A warning is raised in that case instead of returning
    a silently uninformative signal. (Attention dropout inside nn.MultiheadAttention is a
    float, not a module, and is not switched on here.)"""
    if torch is None:
        raise ImportError('mc_dropout_probs requires PyTorch.')
    import warnings
    model.eval()
    n_dropout = _enable_dropout(model)
    if n_dropout == 0:
        warnings.warn('mc_dropout_probs: model has no active (p>0) nn.Dropout modules; MC Dropout uncertainty will be zero.')
    outs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(inputs)
            outs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    model.eval()
    return np.stack(outs, axis=0)

def ensemble_probs(member_probs: list[np.ndarray]) -> np.ndarray:
    return np.stack(member_probs, axis=0)
