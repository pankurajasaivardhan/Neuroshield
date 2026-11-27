# Track 1 reliability evaluation + gate + fingerprint (thresholds from val).
from __future__ import annotations
import argparse
import json
import os
import hashlib
import subprocess
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from shared.reliability.calibration import expected_calibration_error, brier_score, reliability_table, TemperatureScaler
from shared.reliability.uncertainty import predictive_entropy, mutual_information, mc_dropout_probs
from shared.reliability.ood import MahalanobisOOD, ood_overfit_ratio
from shared.reliability.disagreement import classification_disagreement
from shared.reliability import gate as gate_mod
from shared.reliability.fingerprint import FailureFingerprint, FailureCase
from shared.reliability.report import ReliabilityReport
from shared.reliability.scan_eval import run_scan_level
from .data import OASISDataset
from . import train_model_a, train_model_b

def load_checkpoint(model, path, device):
    try:
        obj = torch.load(path, map_location=device)
    except Exception:
        obj = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(obj, dict):
        obj = obj.state_dict()
    if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
        obj = obj['state_dict']
    model.load_state_dict(obj)

@torch.no_grad()
def collect_logits_and_features(model, dataset, device, feature_hook_layer):
    feats = []

    def hook(_m, _in, out):
        feats.append(out.detach().cpu().flatten(1))
    h = feature_hook_layer.register_forward_hook(hook)
    model.eval()
    (logits_all, labels_all, ids_all) = ([], [], [])
    for i in range(len(dataset)):
        (img, label, sid) = dataset[i]
        logits = model(img.unsqueeze(0).to(device))
        logits_all.append(logits[0].cpu().numpy())
        labels_all.append(int(label))
        ids_all.append(sid)
    h.remove()
    return (np.stack(logits_all), np.concatenate([f.numpy() for f in feats]), np.array(labels_all), ids_all)

def vit_feature_layer(model):

    return _TapLayer(model.encoder.ln)

class _TapLayer:

    def __init__(self, module):
        self.module = module

    def register_forward_hook(self, hook):

        def wrapped(m, i, o):
            hook(m, i, o.mean(dim=1) if o.dim() == 3 else o)
        return self.module.register_forward_hook(wrapped)

def resnet_feature_layer(model):
    return model.avgpool

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--model-a', default=None, help='Model A checkpoint path')
    ap.add_argument('--model-b', default=None, help='Model B checkpoint path')
    ap.add_argument('--out', required=True)
    ap.add_argument('--mc-passes', type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import shared.reliability.gate as _g
    def _sha(p):
        if not p: return None
        h = hashlib.sha256()
        with open(p, 'rb') as fh:
            for ch in iter(lambda: fh.read(1 << 20), b''):
                h.update(ch)
        return h.hexdigest()
    try:
        commit = subprocess.run(['git', '-C', os.path.dirname(_g.__file__), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = 'unknown'
    prov = {'git_commit': commit, 'gate_module_path': _g.__file__, 'gate_modes_run': ['legacy', 'v2_target'],
            'checkpoints': {'model_a': {'requested': args.model_a, 'sha256': _sha(args.model_a)}, 'model_b': {'requested': args.model_b, 'sha256': _sha(args.model_b)}}}
    print('PROVENANCE', json.dumps(prov, indent=1), flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    splits = {s: os.path.join(args.data_root, f'splits/{s}.csv') for s in ('train', 'val', 'test')}
    models = {}
    if args.model_a:
        m = train_model_a.build_model()
        load_checkpoint(m, args.model_a, device)
        models['model_a'] = (m.to(device), resnet_feature_layer)
    if args.model_b:
        m = train_model_b.build_model()
        load_checkpoint(m, args.model_b, device)
        models['model_b'] = (m.to(device), vit_feature_layer)
    results = {'models': {}, 'provenance': prov}
    fitted = {}
    for (name, (model, feat_fn)) in models.items():
        per_split = {}
        ood_model = None
        ood_lw = None
        scaler = None
        split_ood = {}
        for split in ('train', 'val', 'test'):
            ds = OASISDataset(splits[split])
            (logits, feats, labels, ids) = collect_logits_and_features(model, ds, device, feat_fn(model))
            if split == 'train':
                ood_model = MahalanobisOOD()
                ood_model.fit(feats, labels)
                ood_lw = MahalanobisOOD(shrinkage='ledoit_wolf')
                ood_lw.fit(feats, labels)
            if split == 'val':
                scaler = TemperatureScaler()
                scaler.fit(logits, labels)
            probs_raw = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            probs_cal = scaler.transform(logits) if scaler else probs_raw
            ood_scores = ood_model.score(feats) if ood_model else np.zeros(len(labels))
            lw_scores = ood_lw.score(feats) if ood_lw else np.zeros(len(labels))
            split_ood[split] = (ood_scores, lw_scores)
            per_split[split] = {'mean_ood_ledoit_wolf': float(lw_scores.mean()), 'ece_raw': expected_calibration_error(probs_raw, labels), 'ece_calibrated': expected_calibration_error(probs_cal, labels), 'brier': brier_score(probs_cal, labels), 'accuracy': float(accuracy_score(labels, probs_cal.argmax(axis=1))), 'mean_ood': float(ood_scores.mean())}
        per_split['ood_diagnostics'] = {
            'legacy_val_over_train': ood_overfit_ratio(split_ood['train'][0], split_ood['val'][0]),
            'legacy_test_over_train': ood_overfit_ratio(split_ood['train'][0], split_ood['test'][0]),
            'ledoit_wolf_val_over_train': ood_overfit_ratio(split_ood['train'][1], split_ood['val'][1]),
            'ledoit_wolf_test_over_train': ood_overfit_ratio(split_ood['train'][1], split_ood['test'][1]),
            'legacy_condition_number': ood_model.condition_number,
            'ledoit_wolf_condition_number': ood_lw.condition_number,
            'ledoit_wolf_shrinkage_intensity': ood_lw.shrinkage_intensity,
            'note': 'descriptive only: train distances are in-sample by construction, so a held-out/train ratio above 1 is expected and is not a transfer or detection verdict; conditioning is not proof of detection ability'}
        fitted[name] = {'ood': ood_model, 'ood_lw': ood_lw, 'scaler': scaler}
        results['models'][name] = per_split
        print(f'[{name}]', json.dumps(per_split, indent=2), flush=True)
    uncertainty = {}
    for (name, (model, _)) in models.items():
        ds = OASISDataset(splits['test'])
        sample_probs = []
        for i in range(len(ds)):
            (img, _, _) = ds[i]
            sample_probs.append(mc_dropout_probs(model, img.unsqueeze(0).to(device), n_passes=args.mc_passes)[..., 0, :])
        sp = np.stack(sample_probs, axis=1)
        uncertainty[name] = {'predictive_entropy': predictive_entropy(sp.mean(axis=0)).tolist(), 'mutual_information': mutual_information(sp).tolist()}
    if len(models) == 2:
        prob_sets = []
        for (name, (model, _)) in models.items():
            ds = OASISDataset(splits['test'])
            (logits, _, labels, ids) = collect_logits_and_features(model, ds, device, resnet_feature_layer(model) if name == 'model_a' else vit_feature_layer(model))
            prob_sets.append(torch.softmax(torch.tensor(logits), dim=-1).numpy())
        dis = classification_disagreement(prob_sets)
        results['disagreement_test'] = {'mismatch_fraction': dis['mismatch_fraction'], 'mean_js_divergence': dis['mean_js_divergence'].tolist()}
    name0 = next(iter(models))
    (model, feat_fn) = models[name0]
    gate_data = {}
    for split in ('val', 'test'):
        ds = OASISDataset(splits[split])
        (logits, feats, labels, ids) = collect_logits_and_features(model, ds, device, feat_fn(model))
        scaler = fitted[name0]['scaler']
        probs = scaler.transform(logits) if scaler else torch.softmax(torch.tensor(logits), dim=-1).numpy()
        conf = probs.max(axis=1)
        ent = predictive_entropy(probs)
        ood = fitted[name0]['ood'].score(feats)
        ood_lw_scores = fitted[name0]['ood_lw'].score(feats)
        errors = (probs.argmax(axis=1) != labels).astype(float)
        gate_data[split] = dict(conf=conf, ent=ent, ood=ood, ood_lw=ood_lw_scores, errors=errors, labels=labels, ids=ids, probs=probs)
    val_risk = gate_mod.risk_score(gate_data['val']['conf'], gate_data['val']['ent'], gate_data['val']['ood'])
    thresholds = gate_mod.select_thresholds(val_risk, gate_data['val']['errors'])
    test_risk = gate_mod.risk_score(gate_data['test']['conf'], gate_data['test']['ent'], gate_data['test']['ood'])
    signals = [{'confidence': float(c), 'entropy': float(e), 'ood': float(o)} for (c, e, o) in zip(gate_data['test']['conf'], gate_data['test']['ent'], gate_data['test']['ood'])]
    decisions = gate_mod.decide(gate_data['test']['ids'], test_risk, thresholds.max_uncertainty, signals)
    curve = gate_mod.risk_coverage_curve(test_risk, gate_data['test']['errors'])
    results['gate'] = {'thresholds': thresholds.__dict__, 'decisions': [d.__dict__ for d in decisions], 'risk_coverage': curve[::max(1, len(curve) // 20)], 'test_error_ungated': float(gate_data['test']['errors'].mean())}
    # Gate v2 (validation-only): fitted normaliser + target-error objective, Ledoit-Wolf OOD.
    gd_v, gd_t = gate_data['val'], gate_data['test']
    norm = gate_mod.RiskNormalizer.fit(gd_v['ent'], gd_v['ood_lw'])
    v2_val_risk = norm.risk(gd_v['conf'], gd_v['ent'], gd_v['ood_lw'])
    v2_test_risk = norm.risk(gd_t['conf'], gd_t['ent'], gd_t['ood_lw'])
    th2 = gate_mod.select_thresholds_target(v2_val_risk, gd_v['errors'])
    dec2 = gate_mod.decide(gd_t['ids'], v2_test_risk, th2.max_uncertainty, signals)
    acc2 = np.array([d.decision.startswith('ACCEPT') for d in dec2])
    results['gate_v2'] = {'thresholds': th2.__dict__, 'n_test': int(len(acc2)), 'coverage_test': float(acc2.mean()),
                          'accepted_error_test': float(gd_t['errors'][acc2].mean()) if acc2.any() else None,
                          'ungated_error_test': float(gd_t['errors'].mean()),
                          'note': 'test numbers are evaluated once with a validation-fitted cutoff; slices of one scan are not independent'}
    results['gate']['legacy_target_note'] = 'legacy target accepted error = median of binary validation errors; with ungated error below 0.5 that median is 0, so the legacy rule almost never accepts a cutoff below 1.0'
    results['slice_counts'] = {sp: int(len(OASISDataset(splits[sp]))) for sp in ('train', 'val', 'test')}
    results['scan_level'] = {}
    for (name, (mdl, ffn)) in models.items():
        d = {}
        for sp in ('val', 'test'):
            ds = OASISDataset(splits[sp])
            (lg, ft, lb, idl) = collect_logits_and_features(mdl, ds, device, ffn(mdl))
            sc = fitted[name]['scaler']
            pr = sc.transform(lg) if sc else torch.softmax(torch.tensor(lg), dim=-1).numpy()
            d[sp] = dict(probs=pr, ood_lw=fitted[name]['ood_lw'].score(ft), ids=idl, labels=lb)
        results['scan_level'][name] = run_scan_level(d['val'], d['test'])
    print('GATE_LEGACY', json.dumps(results['gate']['thresholds']), flush=True)
    print('GATE_V2', json.dumps(results['gate_v2']), flush=True)
    print('SLICE_COUNTS', json.dumps(results['slice_counts']), flush=True)
    print('SCAN_LEVEL', json.dumps(results['scan_level'], indent=1, default=str), flush=True)
    overconf = []
    for (cid, p, lab, r) in zip(gate_data['test']['ids'], gate_data['test']['probs'], gate_data['test']['labels'], test_risk):
        pred = int(p.argmax())
        if pred != lab and p.max() > 0.8:
            overconf.append(FailureCase(case_id=cid, condition='original', error_description=f'predicted class {pred}, true {lab}, confidence {p.max():.3f}', metrics={'confidence': float(p.max()), 'risk': float(r)}))
    if overconf:
        fp = FailureFingerprint(track='track1_alzheimers', model=name0, failure_category='overconfident_misclassification', description=f'{len(overconf)} test cases misclassified with confidence > 0.8.', evidence_cases=overconf, aggregate_signals={'n_cases': len(overconf), 'test_error': float(gate_data['test']['errors'].mean())})
        fp.to_json(os.path.join(args.out, 'failure_fingerprint_overconfident.json'))
        results['failure_fingerprint'] = fp.id
    with open(os.path.join(args.out, 'reliability_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved to', args.out)
if __name__ == '__main__':
    main()
