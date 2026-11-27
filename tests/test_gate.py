"""SYNTHETIC tests of gate code behaviour. No OASIS data is used; these do not reproduce
or replace any reported result."""
import numpy as np
from shared.reliability import gate


def _make(rng, n=175, err=0.39, sep=0.6):
    e = (rng.random(n) < err).astype(float)
    risk = 1 / (1 + np.exp(-(sep * (e - 0.4) + rng.normal(size=n) * 0.8)))
    return risk, e


def test_legacy_target_is_zero_for_binary_errors_below_half():
    e = (np.random.default_rng(0).random(175) < 0.39).astype(float)
    assert np.median(e) == 0.0


def test_legacy_gate_degenerates_with_realistic_ranking():
    rng = np.random.default_rng(1)
    degenerate = 0
    for _ in range(100):
        risk, e = _make(rng)
        degenerate += gate.select_thresholds_legacy(risk, e).max_uncertainty == 1.0
    assert degenerate >= 85  # near-always, whatever the cohort noise


def test_target_gate_finds_cutoff_when_ranking_has_signal():
    rng = np.random.default_rng(2)
    fitted = 0
    for _ in range(100):
        risk, e = _make(rng, sep=1.5)
        th = gate.select_thresholds_target(risk, e)
        if not th.degenerate:
            acc = risk <= th.max_uncertainty
            assert e[acc].mean() <= e.mean() * 0.75 + 1e-9  # validation target honoured
            assert acc.mean() >= 0.15
            fitted += 1
    assert fitted >= 60


def test_target_gate_reports_degenerate_when_ranking_is_noise():
    rng = np.random.default_rng(3)
    risk = rng.random(175)
    e = (rng.random(175) < 0.39).astype(float)
    th = gate.select_thresholds_target(risk, e, conservative=True)
    assert th.degenerate and th.max_uncertainty == 1.0


def test_normalizer_is_fitted_on_validation_not_test():
    rng = np.random.default_rng(4)
    v_u, v_o = rng.random(100), rng.random(100)
    norm = gate.RiskNormalizer.fit(v_u, v_o)
    t_u = rng.random(50) + 5.0  # shifted test uncertainty
    r_shift = norm.risk(np.full(50, 0.9), t_u, rng.random(50))
    r_self = gate.risk_score(np.full(50, 0.9), t_u, rng.random(50))
    assert r_shift.mean() > r_self.mean()  # shift is visible, not rescaled away


if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            v(); print('ok', k)
