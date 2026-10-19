# Model disagreement signals for classification and segmentation.
from __future__ import annotations
import numpy as np

def classification_disagreement(prob_sets: list[np.ndarray]) -> dict:
    n = prob_sets[0].shape[0]
    preds = [p.argmax(axis=1) for p in prob_sets]
    label_mismatch = np.zeros(n, dtype=bool)
    js_sum = np.zeros(n)
    pairs = 0
    for i in range(len(prob_sets)):
        for j in range(i + 1, len(prob_sets)):
            label_mismatch |= preds[i] != preds[j]
            (p, q) = (prob_sets[i], prob_sets[j])
            m = 0.5 * (p + q)
            kl_pm = np.sum(p * np.log(np.clip(p, 1e-12, 1) / np.clip(m, 1e-12, 1)), axis=1)
            kl_qm = np.sum(q * np.log(np.clip(q, 1e-12, 1) / np.clip(m, 1e-12, 1)), axis=1)
            js_sum += 0.5 * (kl_pm + kl_qm)
            pairs += 1
    return {'mean_js_divergence': js_sum / max(pairs, 1), 'label_mismatch': label_mismatch, 'mismatch_fraction': float(label_mismatch.mean())}

def segmentation_disagreement(mask_a: np.ndarray, mask_b: np.ndarray, num_classes: int=4) -> dict:
    union = (mask_a > 0) | (mask_b > 0)
    disagree_voxels = ((mask_a != mask_b) & union).sum()
    union_voxels = int(union.sum())
    per_class_iou = {}
    for c in range(1, num_classes):
        inter = ((mask_a == c) & (mask_b == c)).sum()
        uni = ((mask_a == c) | (mask_b == c)).sum()
        per_class_iou[c] = float(inter / uni) if uni > 0 else None
    return {'foreground_disagreement_rate': float(disagree_voxels / union_voxels) if union_voxels else 0.0, 'per_class_iou': per_class_iou}
