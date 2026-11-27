"""SYNTHETIC checks of the interval helpers in scripts/bootstrap_cis.py (units and behaviour)."""
import csv, importlib.util, os, io, contextlib
import numpy as np
spec = importlib.util.spec_from_file_location('bc', os.path.join(os.path.dirname(__file__), '..', 'scripts', 'bootstrap_cis.py'))
def load():
    bc = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(bc)
    return bc
bc = load()

def test_wilson_known_value():
    p, lo, hi, n = bc.wilson(26, 36)
    assert abs(p - 26 / 36) < 1e-12 and 0.55 < lo < 0.57 and 0.83 < hi < 0.85 and n == 36

def test_boot_width_shrinks_with_n():
    rng = np.random.default_rng(1)
    w = []
    for n in (20, 200):
        x = rng.normal(0, 1, n); _, lo, hi, _ = bc.boot(x, n=2000); w.append(hi - lo)
    assert w[1] < w[0]

def test_output_has_no_slice_level_rows_and_has_units():
    rows = list(csv.DictReader(open(os.path.join(os.path.dirname(__file__), '..', 'docs', 'results', 'bootstrap_cis.csv'))))
    assert not any('JS' in r['quantity'] for r in rows)
    assert all(r['n'] for r in rows) and any('26/36' in r['quantity'] for r in rows)

if __name__ == '__main__':
    test_wilson_known_value(); test_boot_width_shrinks_with_n(); test_output_has_no_slice_level_rows_and_has_units(); print('bootstrap tests OK')
