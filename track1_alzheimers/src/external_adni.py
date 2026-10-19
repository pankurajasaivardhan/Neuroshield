# ADNI external validation (requires ADNI data access).
from __future__ import annotations
import argparse
import json
import torch
from .data import OASISDataset, scan_level_predictions
from . import train_model_a, train_model_b
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adni-csv', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--model', choices=['a', 'b', 'c'], default='b')
    ap.add_argument('--out', default='adni_external_results.json')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    builder = {'a': train_model_a.build_model, 'b': train_model_b.build_model, 'c': train_model_b.build_model}[args.model]
    model = builder()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    ds = OASISDataset(args.adni_csv)
    per_scan = scan_level_predictions(model, ds, device)
    import numpy as np
    y = np.array([v['label'] for v in per_scan.values()])
    p = np.stack([v['probs'] for v in per_scan.values()])
    res = {'cohort': 'ADNI (external)', 'n_scans': int(len(y)), 'accuracy': float(accuracy_score(y, p.argmax(axis=1))), 'balanced_accuracy': float(balanced_accuracy_score(y, p.argmax(axis=1))), 'roc_auc': float(roc_auc_score(y, p[:, 1])) if len(set(y.tolist())) > 1 else None, 'note': 'Frozen OASIS-trained model applied to ADNI without any tuning. Compare against OASIS test metrics to measure domain-shift degradation.'}
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
if __name__ == '__main__':
    main()
