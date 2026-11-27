"""Scan-level selective classification utilities (protocol v2, 2026-11-28).

Pure numpy. Unit of analysis is the scan. Ties are broken by scan id so results are deterministic.
"""
from __future__ import annotations
import math
import numpy as np

COVERAGE_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def aggregate_scans(slice_probs: np.ndarray, scan_ids: list[str], labels: np.ndarray):
    """Mean slice probability per scan (calibration must already be applied to the slice logits).
    Returns (ids sorted, probs [n_scans, C], labels [n_scans])."""
    ids = sorted(set(scan_ids))
    probs, labs = [], []
    arr = np.asarray(scan_ids)
    for sid in ids:
        m = arr == sid
        probs.append(slice_probs[m].mean(axis=0))
        lab = np.unique(labels[m])
        if len(lab) != 1:
            raise ValueError(f'scan {sid} has inconsistent slice labels {lab}')
        labs.append(int(lab[0]))
    return ids, np.stack(probs), np.asarray(labs)


def _order(score_good: np.ndarray, ids: list[str]) -> np.ndarray:
    """Indices from most to least trusted; equal scores ordered by scan id."""
    return np.array(sorted(range(len(ids)), key=lambda i: (-float(score_good[i]), ids[i])))


def accepted_error_at(score_good: np.ndarray, errors: np.ndarray, ids: list[str], coverage: float):
    n = len(ids)
    k = math.ceil(coverage * n)
    if k < 1:
        return float('nan'), 0
    idx = _order(score_good, ids)[:k]
    return float(errors[idx].mean()), k


def risk_coverage(score_good, errors, ids, grid=COVERAGE_GRID):
    return [dict(coverage=c, **dict(zip(('accepted_error', 'n_accepted'), accepted_error_at(score_good, errors, ids, c)))) for c in grid]


def aurc(curve) -> float:
    xs = [r['coverage'] for r in curve]
    ys = [r['accepted_error'] for r in curve]
    return float(sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1)))


def paired_bootstrap(score_a, score_b, errors, ids, n_boot=10000, seed=0, grid=COVERAGE_GRID):
    """Paired bootstrap over scans of (method A - method B) accepted-error at each grid coverage and AURC.
    A scan resampled twice keeps its id with a suffix so ties stay deterministic. Descriptive only."""
    rng = np.random.default_rng(seed)
    n = len(ids)
    diffs = {c: [] for c in grid}
    d_aurc = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, n)
        bid = [f'{ids[j]}#{t}' for t, j in enumerate(pick)]
        ea = errors[pick]
        ca = risk_coverage(score_a[pick], ea, bid, grid)
        cb = risk_coverage(score_b[pick], ea, bid, grid)
        for ra, rb in zip(ca, cb):
            diffs[ra['coverage']].append(ra['accepted_error'] - rb['accepted_error'])
        d_aurc.append(aurc(ca) - aurc(cb))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {'by_coverage': {c: q(v) for c, v in diffs.items()}, 'aurc_diff_ci95': q(d_aurc),
            'note': 'descriptive paired bootstrap over scans; same test split reanalysed; no superiority conclusion'}
