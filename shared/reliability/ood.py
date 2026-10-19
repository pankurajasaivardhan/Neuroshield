# OOD detection: Mahalanobis distance in feature space (fit on train only).
from __future__ import annotations
import numpy as np

class MahalanobisOOD:

    def __init__(self):
        self.class_means: list[np.ndarray] = []
        self.precision: np.ndarray | None = None

    def fit(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
        classes = np.unique(train_labels)
        self.class_means = [train_features[train_labels == c].mean(axis=0) for c in classes]
        centered = np.concatenate([train_features[train_labels == c] - self.class_means[i] for (i, c) in enumerate(classes)], axis=0)
        cov = np.cov(centered, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-06
        self.precision = np.linalg.pinv(cov)

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.precision is None:
            raise RuntimeError('MahalanobisOOD not fitted.')
        dists = []
        for mean in self.class_means:
            diff = features - mean
            d = np.einsum('ij,jk,ik->i', diff, self.precision, diff)
            dists.append(d)
        return np.min(np.stack(dists, axis=1), axis=1)
