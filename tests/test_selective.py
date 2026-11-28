"""SYNTHETIC tests for scan-level selective utilities. They check code behaviour only."""
import numpy as np
import pytest
from shared.reliability import selective as S


def test_aggregate_mean_and_labels():
    p = np.array([[.9, .1], [.7, .3], [.2, .8]])
    ids, probs, labs = S.aggregate_scans(p, ['b', 'b', 'a'], np.array([0, 0, 1]))
    assert ids == ['a', 'b']
    assert np.allclose(probs[1], [.8, .2]) and list(labs) == [1, 0]


def test_aggregate_rejects_mixed_labels():
    with pytest.raises(ValueError):
        S.aggregate_scans(np.ones((2, 2)) / 2, ['a', 'a'], np.array([0, 1]))


def test_perfect_ranking_has_zero_error_until_wrong_scans():
    ids = [f's{i}' for i in range(10)]
    err = np.array([0] * 8 + [1] * 2, float)
    score = np.linspace(1, 0, 10)  # most trusted first, errors last
    rc = S.risk_coverage(score, err, ids)
    assert rc[0]['accepted_error'] == 0.0 and rc[-1]['accepted_error'] == pytest.approx(0.2)
    assert S.aurc(rc) < 0.1


def test_ties_are_deterministic_by_id():
    ids = ['c', 'a', 'b']
    err = np.array([1., 0., 0.])
    e1, k1 = S.accepted_error_at(np.ones(3), err, ids, 0.34)  # k=2: ids a,b accepted
    assert (e1, k1) == (0.0, 2)


def test_random_scores_paired_bootstrap_includes_zero():
    rng = np.random.default_rng(1)
    n = 36
    ids = [f's{i}' for i in range(n)]
    err = (rng.random(n) < 0.4).astype(float)
    a, b = rng.random(n), rng.random(n)
    r = S.paired_bootstrap(a, b, err, ids, n_boot=300, seed=0)
    lo, hi = r['aurc_diff_ci95']
    assert lo < 0 < hi


def test_bootstrap_identical_methods_give_exactly_zero_difference():
    ids = [f's{i}' for i in range(12)]
    err = np.array([0, 1] * 6, float)
    s = np.linspace(0, 1, 12)
    r = S.paired_bootstrap(s, s.copy(), err, ids, n_boot=200, seed=3)
    assert r['aurc_diff_ci95'] == [0.0, 0.0]
    assert all(v == [0.0, 0.0] for v in r['by_coverage'].values())


def test_bootstrap_perfect_vs_reversed_is_negative_at_partial_coverage():
    n = 30
    ids = [f's{i}' for i in range(n)]
    err = np.array([0.] * 20 + [1.] * 10)
    good = np.linspace(1, 0, n)  # errors ranked last
    bad = good[::-1].copy()      # errors ranked first
    r = S.paired_bootstrap(good, bad, err, ids, n_boot=300, seed=1)
    lo, hi = r['aurc_diff_ci95']
    assert hi < 0
    assert r['by_coverage'][1.0] == [0.0, 0.0]  # full coverage: same set accepted, no difference


def test_duplicate_resampled_scans_keep_deterministic_order():
    ids = ['a#0', 'a#1', 'b#2']  # a scan drawn twice has identical score and error
    err = np.array([1., 1., 0.])
    e, k = S.accepted_error_at(np.array([.5, .5, .5]), err, ids, 0.66)
    assert k == 2 and e == 1.0  # a#0, a#1 first by id; result reproducible
