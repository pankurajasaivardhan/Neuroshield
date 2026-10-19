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

def _enable_dropout(model) -> None:
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def mc_dropout_probs(model, inputs, n_passes: int=30) -> np.ndarray:
    if torch is None:
        raise ImportError('mc_dropout_probs requires PyTorch.')
    model.eval()
    _enable_dropout(model)
    outs = []
    for _ in range(n_passes):
        logits = model(inputs)
        with torch.no_grad():
            logits = model(inputs)
        outs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    model.eval()
    return np.stack(outs, axis=0)

def ensemble_probs(member_probs: list[np.ndarray]) -> np.ndarray:
    return np.stack(member_probs, axis=0)
