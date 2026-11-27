"""SYNTHETIC tests of the OOD estimator. They show conditioning and in-sample overfit
behaviour for n < d features and that the score separates a simulated shift. They are not
evidence about the OASIS features or about any real covariate shift."""
import numpy as np
from shared.reliability.ood import MahalanobisOOD, ood_overfit_ratio


def _data(rng, n=400, d=300, shift=0.0, n_test=150):
    tr = rng.normal(size=(n, d)); lab = (rng.random(n) < .5).astype(int)
    ho = rng.normal(size=(n_test, d)) + shift
    return tr, lab, ho


def test_legacy_overfits_when_n_lt_d_and_shrinkage_reduces_it():
    rng = np.random.default_rng(0)
    tr, lab, ho = _data(rng)
    legacy = MahalanobisOOD(); legacy.fit(tr, lab)
    lw = MahalanobisOOD(shrinkage='ledoit_wolf'); lw.fit(tr, lab)
    r_legacy = ood_overfit_ratio(legacy.score(tr), legacy.score(ho))
    r_lw = ood_overfit_ratio(lw.score(tr), lw.score(ho))
    assert r_legacy > 3 and r_lw < r_legacy / 2
    assert lw.condition_number < legacy.condition_number


def test_shrinkage_score_still_separates_a_simulated_shift():
    rng = np.random.default_rng(1)
    tr, lab, ho_id = _data(rng)
    _, _, ho_shift = _data(rng, shift=0.6)
    lw = MahalanobisOOD(shrinkage='ledoit_wolf'); lw.fit(tr, lab)
    a, b = lw.score(ho_id), lw.score(ho_shift)
    r = np.concatenate([a, b]).argsort().argsort() + 1
    auroc = (r[len(a):].sum() - len(b) * (len(b) + 1) / 2) / (len(a) * len(b))
    assert auroc > 0.9


if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            v(); print('ok', k)
