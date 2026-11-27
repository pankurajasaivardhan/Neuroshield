"""Scan-level gate vs confidence-only baseline (protocol v2, 2026-11-28). Descriptive reanalysis, no superiority claim."""
from __future__ import annotations
import math
import numpy as np
from . import selective as S
from .gate import RiskNormalizer
from .uncertainty import predictive_entropy


def _scan_table(probs, ood_lw, ids, labels):
    sid, sp, sl = S.aggregate_scans(probs, ids, labels)
    arr = np.asarray(ids)
    sood = np.array([ood_lw[arr == s].mean() for s in sid])
    return dict(ids=sid, probs=sp, labels=sl, conf=sp.max(axis=1), ent=predictive_entropy(sp), ood=sood,
                errors=(sp.argmax(axis=1) != sl).astype(float))


def _val_operating_point(score_good, errors, ids, floor=0.6):
    rows = [r for r in S.risk_coverage(score_good, errors, ids) if r['coverage'] >= floor]
    best = min(r['accepted_error'] for r in rows)
    cov = min(r['coverage'] for r in rows if r['accepted_error'] == best)
    k = math.ceil(cov * len(ids))
    cutoff = float(score_good[S._order(score_good, ids)[k - 1]])
    return cov, cutoff, best


def run_scan_level(val, test):
    """val/test: dicts with probs (calibrated slice probs), ood_lw (slice), ids (slice->scan id), labels."""
    tv = _scan_table(val['probs'], val['ood_lw'], val['ids'], val['labels'])
    tt = _scan_table(test['probs'], test['ood_lw'], test['ids'], test['labels'])
    norm = RiskNormalizer.fit(tv['ent'], tv['ood'])
    def good_gate(t): return 1.0 - norm.risk(t['conf'], t['ent'], t['ood'])
    def good_conf(t): return t['conf']
    out = {'unit': 'scan', 'n_val': len(tv['ids']), 'n_test': len(tt['ids']),
           'n_val_slices': int(len(val['ids'])), 'n_test_slices': int(len(test['ids'])),
           'aggregation': 'validation-fitted temperature -> softmax -> mean per scan; OOD = mean slice Ledoit-Wolf score',
           'scan_accuracy_test': float(1 - tt['errors'].mean()), 'scan_accuracy_val': float(1 - tv['errors'].mean()),
           'methods': {}}
    for name, fn in (('gate_v2_scan', good_gate), ('confidence_only', good_conf)):
        gv, gt = fn(tv), fn(tt)
        cov, cut, verr = _val_operating_point(gv, tv['errors'], tv['ids'])
        acc = gt >= cut
        curve = S.risk_coverage(gt, tt['errors'], tt['ids'])
        out['methods'][name] = {
            'val_curve': S.risk_coverage(gv, tv['errors'], tv['ids']), 'test_curve': curve, 'test_aurc': S.aurc(curve),
            'val_operating_point': {'coverage': cov, 'cutoff': cut, 'val_accepted_error': verr,
                                    'test_coverage': float(acc.mean()), 'test_n_accepted': int(acc.sum()),
                                    'test_accepted_error': float(tt['errors'][acc].mean()) if acc.any() else None}}
    out['paired_bootstrap_gate_minus_conf'] = S.paired_bootstrap(good_gate(tt), good_conf(tt), tt['errors'], tt['ids'])
    out['note'] = 'descriptive; same test split reanalysed; single seed 42; no superiority conclusion'
    return out
