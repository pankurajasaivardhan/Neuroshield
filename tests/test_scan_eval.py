import numpy as np
from shared.reliability.scan_eval import run_scan_level

def _split(n_scans, rng, spc=5):
    ids, probs, ood, labels = [], [], [], []
    for s in range(n_scans):
        y = s % 2
        ok = rng.random() < 0.7
        p1 = 0.9 if (y == 1) == ok else 0.1
        for _ in range(spc):
            q = np.clip(p1 + rng.normal(0, .02), .01, .99)
            ids.append(f's{s:03d}'); probs.append([1 - q, q]); ood.append(rng.random()); labels.append(y)
    return dict(ids=ids, probs=np.array(probs), ood_lw=np.array(ood), labels=np.array(labels))

def test_runs_and_counts():
    rng = np.random.default_rng(1)
    r = run_scan_level(_split(35, rng), _split(36, rng))
    assert r['n_test'] == 36 and r['n_test_slices'] == 180 and r['n_val'] == 35
    for m in r['methods'].values():
        assert len(m['test_curve']) == 6
    assert 'by_coverage' in r['paired_bootstrap_gate_minus_conf']

def test_full_coverage_equal_error():
    rng = np.random.default_rng(2)
    r = run_scan_level(_split(35, rng), _split(36, rng))
    a = r['methods']['gate_v2_scan']['test_curve'][-1]['accepted_error']
    b = r['methods']['confidence_only']['test_curve'][-1]['accepted_error']
    assert a == b
