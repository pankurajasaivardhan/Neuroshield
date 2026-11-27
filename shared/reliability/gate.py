# Reliability Gate: ACCEPT/ABSTAIN decision + risk-coverage evaluation.
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class GateThresholds:
    max_uncertainty: float
    max_ood: float
    min_confidence: float
    source: str = 'validation'
    degenerate: bool = False

@dataclass
class GateDecision:
    case_id: str
    decision: str
    reasons: list[str]
    signals: dict

def _normalize(x: np.ndarray) -> np.ndarray:
    (lo, hi) = np.percentile(x, [5, 95])
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 2)

def risk_score(confidence: np.ndarray, uncertainty: np.ndarray, ood: np.ndarray, disagreement: np.ndarray | None=None) -> np.ndarray:
    parts = [_normalize(uncertainty), _normalize(ood), 1.0 - confidence]
    if disagreement is not None:
        parts.append(_normalize(disagreement.astype(float)))
    return np.clip(np.mean(parts, axis=0), 0, 1)

def select_thresholds_legacy(val_risk: np.ndarray, val_errors: np.ndarray, target_accepted_error: float | None=None) -> GateThresholds:
    base_error = float(val_errors.mean())
    target = target_accepted_error if target_accepted_error is not None else np.median(val_errors)
    best_cut = 1.0
    for cut in np.linspace(0.05, 0.95, 19):
        accepted = val_risk <= cut
        if accepted.sum() < max(5, int(0.1 * len(val_risk))):
            continue
        err = float(val_errors[accepted].mean())
        if err <= target and err < base_error:
            best_cut = cut
    return GateThresholds(max_uncertainty=best_cut, max_ood=float('inf'), min_confidence=0.0, source=f'validation risk cutoff={best_cut:.2f}, ungated_error={base_error:.3f}')

def decide(case_ids: list[str], risk: np.ndarray, cut: float, signals: list[dict]) -> list[GateDecision]:
    out = []
    for (cid, r, sig) in zip(case_ids, risk, signals):
        reasons = []
        if r > cut:
            reasons.append(f'combined risk {r:.3f} above validation-derived cutoff {cut:.3f}')
        out.append(GateDecision(case_id=cid, decision='ABSTAIN_HUMAN_REVIEW' if reasons else 'ACCEPT_FOR_RESEARCH_EVALUATION', reasons=reasons, signals=sig))
    return out

def risk_coverage_curve(risk: np.ndarray, errors: np.ndarray) -> list[dict]:
    rows = []
    order = np.argsort(risk)
    sorted_err = errors[order]
    n = len(risk)
    for k in range(1, n + 1):
        rows.append({'coverage': k / n, 'accepted_error_rate': float(sorted_err[:k].mean())})
    return rows


# select_thresholds_legacy above is the original objective, kept unchanged so the
# reported Track 1 run stays reproducible. Its target is the MEDIAN of the binary
# validation errors, which is 0 whenever the error rate is below 50%, so a cutoff
# qualifies only if the accepted set has zero errors (see tests/test_gate.py).
select_thresholds = select_thresholds_legacy


@dataclass
class RiskNormalizer:
    """5th/95th percentile scaling fitted on VALIDATION signals, then applied unchanged
    to any split, so a validation cutoff means the same thing on test data."""
    bounds: dict

    @staticmethod
    def fit(uncertainty: np.ndarray, ood: np.ndarray, disagreement: np.ndarray | None = None) -> 'RiskNormalizer':
        b = {'uncertainty': [float(v) for v in np.percentile(uncertainty, [5, 95])],
             'ood': [float(v) for v in np.percentile(ood, [5, 95])]}
        if disagreement is not None:
            b['disagreement'] = [float(v) for v in np.percentile(disagreement.astype(float), [5, 95])]
        return RiskNormalizer(b)

    def _n(self, x: np.ndarray, key: str) -> np.ndarray:
        lo, hi = self.bounds[key]
        if hi - lo < 1e-12:
            return np.zeros_like(x, dtype=float)
        return np.clip((x - lo) / (hi - lo), 0, 2)

    def risk(self, confidence: np.ndarray, uncertainty: np.ndarray, ood: np.ndarray,
             disagreement: np.ndarray | None = None) -> np.ndarray:
        parts = [self._n(uncertainty, 'uncertainty'), self._n(ood, 'ood'), 1.0 - confidence]
        if disagreement is not None and 'disagreement' in self.bounds:
            parts.append(self._n(disagreement.astype(float), 'disagreement'))
        return np.clip(np.mean(parts, axis=0), 0, 1)


def wilson_upper(errors_k: float, n: int, z: float = 1.2816) -> float:
    """One-sided ~90% Wilson upper bound on an error rate."""
    if n == 0:
        return 1.0
    p = errors_k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(min(1.0, (centre + margin) / denom))


def select_thresholds_target(val_risk: np.ndarray, val_errors: np.ndarray,
                             relative_reduction: float = 0.25,
                             target_error: float | None = None,
                             min_accepted_frac: float = 0.15,
                             conservative: bool = False) -> GateThresholds:
    """Pick the cutoff with the largest coverage whose accepted VALIDATION error meets a
    target (default: base error reduced by `relative_reduction`). Validation data only.
    If no cutoff meets the target the gate is reported as degenerate (cutoff 1.0), never
    silently as a working gate. conservative=True compares a Wilson upper bound."""
    base = float(val_errors.mean())
    target = target_error if target_error is not None else base * (1.0 - relative_reduction)
    n = len(val_risk)
    min_n = max(5, int(np.ceil(min_accepted_frac * n)))
    best_cut, best_cov = None, -1.0
    for cut in np.unique(np.quantile(val_risk, np.linspace(0.05, 1.0, 40))):
        acc = val_risk <= cut
        k = int(acc.sum())
        if k < min_n:
            continue
        err = float(val_errors[acc].mean())
        stat = wilson_upper(float(val_errors[acc].sum()), k) if conservative else err
        if stat <= target and k > best_cov:
            best_cut, best_cov = float(cut), k
    if best_cut is None:
        return GateThresholds(max_uncertainty=1.0, max_ood=float('inf'), min_confidence=0.0,
                              source=f'validation: no cutoff met target accepted error {target:.3f} (ungated {base:.3f}); gate degenerate',
                              degenerate=True)
    return GateThresholds(max_uncertainty=best_cut, max_ood=float('inf'), min_confidence=0.0,
                          source=f'validation risk cutoff={best_cut:.3f}, coverage={best_cov / n:.2f}, target_error={target:.3f}, ungated_error={base:.3f}',
                          degenerate=False)
