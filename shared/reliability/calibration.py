# Calibration: ECE, Brier score, temperature scaling (fit on validation only).
from __future__ import annotations
import numpy as np

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int=15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    ece = 0.0
    for b in range(n_bins):
        (lo, hi) = (b / n_bins, (b + 1) / n_bins)
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    (n, c) = probs.shape
    onehot = np.zeros((n, c))
    onehot[np.arange(n), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

def reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int=10) -> list[dict]:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    rows = []
    for b in range(n_bins):
        (lo, hi) = (b / n_bins, (b + 1) / n_bins)
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        rows.append({'bin': f'({lo:.2f}, {hi:.2f}]', 'count': int(mask.sum()), 'mean_confidence': float(confidences[mask].mean()), 'accuracy': float(accuracies[mask].mean())})
    return rows

class TemperatureScaler:

    def __init__(self):
        self.temperature: float | None = None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        from scipy.optimize import minimize_scalar

        def nll(t):
            t = max(t, 0.001)
            scaled = logits / t
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
            return float(-log_probs[np.arange(len(labels)), labels].mean())
        result = minimize_scalar(nll, bounds=(0.05, 50.0), method='bounded')
        self.temperature = float(result.x)
        return self.temperature

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature is None:
            raise RuntimeError('TemperatureScaler not fitted. Call fit() on validation data first.')
        scaled = logits / self.temperature
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        return exp / exp.sum(axis=1, keepdims=True)
