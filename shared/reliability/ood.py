# OOD detection: Mahalanobis distance in feature space (fit on train only).
from __future__ import annotations
import numpy as np


def ledoit_wolf_shrinkage(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage of the sample covariance towards mu*I (x: n x d, already centred).
    Returns (shrunk covariance, shrinkage intensity). Plain numpy, no sklearn needed."""
    n, d = x.shape
    emp = (x.T @ x) / n
    mu = np.trace(emp) / d
    delta = np.sum((emp - mu * np.eye(d)) ** 2) / d
    x2 = x ** 2
    beta_ = (np.sum((x2.T @ x2)) / n - np.sum(emp ** 2)) / (n * d)
    beta = min(max(beta_, 0.0), delta)
    shrink = 0.0 if delta == 0 else beta / delta
    return (1 - shrink) * emp + shrink * mu * np.eye(d), float(shrink)


class MahalanobisOOD:
    """Class-conditional Gaussians with a tied covariance.

    shrinkage='none' reproduces the original estimator (sample covariance + 1e-6*I, pinv),
    which is what the reported Track 1 run used. shrinkage='ledoit_wolf' is a better
    conditioned estimator for n < d feature sets. It improves numerical conditioning only;
    whether the resulting score detects shifted data must be checked on a held-out shifted
    set (see ood_overfit_ratio and tests/test_ood.py for the synthetic checks)."""

    def __init__(self, shrinkage: str = 'none'):
        if shrinkage not in ('none', 'ledoit_wolf'):
            raise ValueError("shrinkage must be 'none' or 'ledoit_wolf'")
        self.shrinkage = shrinkage
        self.class_means: list[np.ndarray] = []
        self.precision: np.ndarray | None = None
        self.shrinkage_intensity: float | None = None
        self.condition_number: float | None = None

    def fit(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
        train_features = train_features.reshape(train_features.shape[0], -1)
        classes = np.unique(train_labels)
        self.class_means = [train_features[train_labels == c].mean(axis=0) for c in classes]
        centered = np.concatenate([train_features[train_labels == c] - self.class_means[i] for (i, c) in enumerate(classes)], axis=0)
        if self.shrinkage == 'ledoit_wolf':
            cov, self.shrinkage_intensity = ledoit_wolf_shrinkage(centered)
        else:
            cov = np.cov(centered, rowvar=False)
        cov = cov + np.eye(cov.shape[0]) * 1e-06
        ev = np.linalg.eigvalsh(cov)
        self.condition_number = float(ev.max() / max(ev.min(), 1e-300))
        self.precision = np.linalg.pinv(cov)

    def score(self, features: np.ndarray) -> np.ndarray:
        features = features.reshape(features.shape[0], -1)
        if self.precision is None:
            raise RuntimeError('MahalanobisOOD not fitted.')
        dists = []
        for mean in self.class_means:
            diff = features - mean
            d = np.einsum('ij,jk,ik->i', diff, self.precision, diff)
            dists.append(d)
        return np.min(np.stack(dists, axis=1), axis=1)


def ood_overfit_ratio(train_scores: np.ndarray, heldout_scores: np.ndarray) -> float:
    """Mean held-out score / mean in-sample score. For a well-conditioned detector fitted on
    in-distribution data this is near 1 on held-out in-distribution data. Values far above 1
    flag an overfit covariance (the reported Track 1 run had ~14x for ViT, ~1e4x for ResNet)."""
    return float(np.mean(heldout_scores) / max(np.mean(train_scores), 1e-12))
