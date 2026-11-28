"""SYNTHETIC contract tests for the changed Track 1 evaluation paths (eval_reliability.main and
repair_model_c.main). They use tiny fake data and a tiny CNN, so they check logic and output
contracts (what is selected, what is fitted on which split, how often test is touched). They say
nothing about performance on OASIS. Run: PYTHONPATH=. python3 tests/test_track1_paths.py"""
import json, os, sys, tempfile, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn

import track1_alzheimers.src.eval_reliability as ER
import track1_alzheimers.src.repair_model_c as RC
from shared.reliability.fingerprint import FailureFingerprint, FailureCase


class FakeDS(torch.utils.data.Dataset):
    def __init__(self, csv_path, expand=True):
        df = pd.read_csv(csv_path)
        self.df = df.loc[df.index.repeat(5)].reset_index(drop=True)  # 5 slices per scan, like the real set
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        g = torch.Generator().manual_seed(int(r['seed']) * 10 + i % 5)
        x = torch.randn(3, 8, 8, generator=g) * 0.5 + (1.0 if r['signal'] else -1.0)
        return x, torch.tensor(int(r['label'])), str(r['ID'])


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 2)
    def forward(self, x):
        return self.fc(self.avgpool(torch.relu(self.conv(x))).flatten(1))


def make_root(tmp, n=(60, 40, 40), flip_train=0):
    os.makedirs(os.path.join(tmp, 'splits'))
    rng = np.random.default_rng(0)
    for name, k in zip(('train', 'val', 'test'), n):
        label = rng.integers(0, 2, k)
        signal = label.copy()
        flips = rng.random(k) < 0.25
        signal = np.where(flips, 1 - signal, signal)  # 25% of scans look like the other class
        df = pd.DataFrame({'ID': [f'{name}_{i}' for i in range(k)], 'label': label, 'signal': signal,
                           'seed': np.arange(k) + {'train': 0, 'val': 1000, 'test': 2000}[name], 'scan_path': 'x'})
        df.to_csv(os.path.join(tmp, 'splits', f'{name}.csv'), index=False)


def trained_ckpt(tmp, root):
    torch.manual_seed(0)
    m = Tiny(); opt = torch.optim.Adam(m.parameters(), lr=0.05)
    ds = FakeDS(os.path.join(root, 'splits', 'train.csv'))
    for _ in range(40):
        for i in range(0, len(ds), 25):
            b = [ds[j] for j in range(i, min(i + 25, len(ds)))]
            x = torch.stack([t[0] for t in b]); y = torch.stack([t[1] for t in b])
            opt.zero_grad(); nn.functional.cross_entropy(m(x), y).backward(); opt.step()
    p = os.path.join(tmp, 'base.pt'); torch.save(m.state_dict(), p); return p


def test_eval_reliability():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, 'root'); os.makedirs(root); make_root(root)
        ck = trained_ckpt(tmp, root); out = os.path.join(tmp, 'out')
        ER.OASISDataset = FakeDS; ER.train_model_a.build_model = Tiny
        sys.argv = ['x', '--data-root', root, '--model-a', ck, '--out', out, '--mc-passes', '2']
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always'); ER.main()
        res = json.load(open(os.path.join(out, 'reliability_results.json')))
        assert any('dropout' in str(x.message).lower() for x in w), 'must warn that MC dropout is inactive'
        m = res['models']['model_a']
        for k in ('train', 'val', 'test', 'ood_diagnostics'): assert k in m
        for k in ('legacy_val_over_train', 'ledoit_wolf_val_over_train', 'legacy_condition_number', 'ledoit_wolf_shrinkage_intensity'):
            assert k in m['ood_diagnostics']
        g2 = res['gate_v2']
        assert g2['n_test'] == 200 and 0.0 <= g2['coverage_test'] <= 1.0
        assert 'degenerate' in g2['thresholds']
        if g2['accepted_error_test'] is not None:
            assert 0.0 <= g2['accepted_error_test'] <= 1.0
        assert abs(g2['ungated_error_test'] - res['gate']['test_error_ungated']) < 1e-9
        assert 'gate' in res and 'risk_coverage' in res['gate']
    print('eval_reliability contract OK  (gate_v2 coverage %.2f)' % g2['coverage_test'])


def run_repair(tmp, root, ck, fp_path, val_losses, force_no_errors=False):
    calls = {'test': 0, 'weights': None}
    RC.OASISDataset = FakeDS
    RC.build_model = Tiny
    RC.scan_level_predictions = __import__('track1_alzheimers.src.data', fromlist=['x']).scan_level_predictions
    RC.collate_fn = lambda b: (torch.stack([t[0] for t in b]), torch.stack([t[1] for t in b]), [t[2] for t in b])
    def fake_eval(model, csv, device):
        calls['test'] += 1; assert csv.endswith('test.csv'); return {'accuracy': 0.5, 'balanced_accuracy': 0.5, 'roc_auc': 0.5}
    RC.evaluate_test = fake_eval
    it = iter(val_losses)
    RC.val_loss_and_scan_acc = lambda model, v, d, c: (next(it), 0.5)
    orig_ws = RC.WeightedRandomSampler
    def spy(weights, **kw):
        calls['weights'] = list(weights); return orig_ws(weights, **kw)
    RC.WeightedRandomSampler = spy
    if force_no_errors:
        RC.find_failing_train_ids = lambda *a, **k: set()
    out = os.path.join(tmp, 'rep')
    argv = ['x', '--data-root', root, '--base-checkpoint', ck, '--out', out, '--epochs', '3', '--conf-threshold', '0.6']
    if fp_path: argv += ['--fingerprint', fp_path]
    sys.argv = argv; RC.main()
    return calls, json.load(open(os.path.join(out, 'model_c_test_results.json')))


def test_repair_selection_and_leak_guard():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, 'root'); os.makedirs(root); make_root(root)
        ck = trained_ckpt(tmp, root)
        fp = FailureFingerprint(track='t1', model='m', failure_category='overconfident_misclassification', description='d',
                                evidence_cases=[FailureCase(case_id='test_3', condition='original', error_description='e', metrics={}),
                                                FailureCase(case_id='test_7', condition='original', error_description='e', metrics={})],
                                aggregate_signals={})
        fpp = os.path.join(tmp, 'fp.json'); fp.to_json(fpp)
        calls, res = run_repair(tmp, root, ck, fpp, [1.0, 0.9, 0.5, 0.7])
        tr = pd.read_csv(os.path.join(root, 'splits', 'train.csv'))
        w = calls['weights']; assert len(w) == len(tr) * 5
        up = {tr.ID.repeat(5).reset_index(drop=True)[i] for i, x in enumerate(w) if x > 1}
        assert up and all(u.startswith('train_') for u in up), 'only train scans may be oversampled'
        assert res['repair']['failing_train_scans'] == len(up) and res['repair']['status'] == 'run'
        assert res['repair']['fingerprint_cases_in_test_split'] == 2, 'test-ID fingerprint must be detected, not used'
        assert res['repair']['best_epoch_by_val_loss'] == 2, 'checkpoint must be chosen by validation loss'
        assert calls['test'] == 1, 'test split evaluated exactly once'
    print('repair selection / leak guard OK  (%d train scans oversampled)' % len(up))


def test_repair_not_run_when_no_errors():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, 'root'); os.makedirs(root); make_root(root)
        ck = trained_ckpt(tmp, root)
        calls, res = run_repair(tmp, root, ck, None, [1.0], force_no_errors=True)
        assert res['repair']['status'] == 'not_run' and calls['test'] == 0
    print('repair not_run path OK (test split untouched)')


if __name__ == '__main__':
    test_eval_reliability(); test_repair_selection_and_leak_guard(); test_repair_not_run_when_no_errors()
