"""Track 1: Alzheimer's classification (OASIS, ViT-B/16 ensemble, 3 real seeds).

Mirrors track1_alzheimers/src/data.py preprocessing exactly:
5 sagittal slices at offsets (-30,-15,0,15,30), non-zero voxel percentile
1-99 normalization, 224x224 bilinear resize, ImageNet mean/std.
"""
from __future__ import annotations

import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16

IMG_SIZE = 224
IMAGENET_MEAN = 0.485
IMAGENET_STD = 0.229
SLICE_OFFSETS = (-30, -15, 0, 15, 30)
SEEDS = (42, 123, 2024)

CLASS_LABELS = {0: "No dementia pattern", 1: "Dementia pattern"}
CLASS_NOTE = (
    "The model sorts a structural MRI into two groups learned from the OASIS dataset: "
    "scans labelled with dementia signs (1) vs without (0). Held-out test accuracy "
    "~71%. A model output, not a diagnosis."
)

_models: dict[int, nn.Module] = {}


def _build_model() -> nn.Module:
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    return model


def _load_one(ckpt_path: str) -> nn.Module:
    model = _build_model()
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                state = obj[key]
                break
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_models(ckpt_dir: str) -> None:
    for seed in SEEDS:
        if seed not in _models:
            path = os.path.join(ckpt_dir, f"model_b_vit_seed{seed}_best.pt")
            _models[seed] = _load_one(path)


def normalize_mri_slice(stack: np.ndarray) -> np.ndarray:
    nonzero = stack[stack > 0]
    if len(nonzero) == 0:
        return stack
    p1, p99 = np.percentile(nonzero, [1, 99])
    stack = np.clip(stack, p1, p99)
    stack = (stack - p1) / (p99 - p1 + 1e-8)
    return np.clip(stack, 0, 1).astype(np.float32)


def volume_to_tensor(volume: np.ndarray) -> torch.Tensor:
    """Replicates OASISDataset.__getitem__ for all 5 offsets -> (5,3,224,224)."""
    volume = np.squeeze(volume)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D scan after squeeze, got shape {volume.shape}")
    center = volume.shape[0] // 2
    slices = []
    for off in SLICE_OFFSETS:
        idx = int(np.clip(center + off, 0, volume.shape[0] - 1))
        s = normalize_mri_slice(volume[idx, :, :].astype(np.float32))
        t = torch.from_numpy(s)[None, None]
        t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]
        t = (t - IMAGENET_MEAN) / IMAGENET_STD
        slices.append(t.repeat(3, 1, 1))
    return torch.stack(slices)


def load_volume(path: str, pair_path: str | None = None) -> np.ndarray:
    img = nib.load(path)
    return img.get_fdata()


def run_inference(scan_path: str) -> dict:
    if not _models:
        raise RuntimeError("Track 1 models not loaded")
    volume = load_volume(scan_path)
    x = volume_to_tensor(volume)  # (5,3,224,224)
    per_seed = {}
    all_probs = []
    with torch.no_grad():
        for seed, model in _models.items():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).numpy()  # (5,2)
            scan_probs = probs.mean(axis=0)
            all_probs.append(scan_probs)
            per_seed[str(seed)] = {
                "prediction": int(scan_probs.argmax()),
                "prob_dementia": round(float(scan_probs[1]), 4),
            }
    mean_probs = np.mean(np.stack(all_probs), axis=0)
    pred = int(mean_probs.argmax())
    preds = [v["prediction"] for v in per_seed.values()]
    agreement = len(set(preds)) == 1
    return {
        "prediction": pred,
        "prediction_label": CLASS_LABELS[pred],
        "probability_dementia": round(float(mean_probs[1]), 4),
        "probability_no_dementia": round(float(mean_probs[0]), 4),
        "seed_agreement": agreement,
        "seeds_agreeing": int(sum(1 for p in preds if p == pred)),
        "seeds_total": len(preds),
        "per_seed": per_seed,
        "class_note": CLASS_NOTE,
        "slices_used": len(SLICE_OFFSETS),
    }
