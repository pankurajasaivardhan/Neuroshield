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

def select_thresholds(val_risk: np.ndarray, val_errors: np.ndarray, target_accepted_error: float | None=None) -> GateThresholds:
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
