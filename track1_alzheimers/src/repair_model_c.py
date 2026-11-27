# Track 1 Model C: targeted repair driven by a Failure Fingerprint.
#
# Selection rule (changed from the first version): the cases that get oversampled are
# found by running the BASE model on the TRAIN split and taking its confident errors.
# The fingerprint's own case IDs are never used to pick training cases, because the
# original fingerprint was built from TEST cases (no test case is in the train split, so
# the old intersection was empty and the repair was a no-op) and reusing test failures
# to choose training data would leak the test set. The checkpoint to keep is chosen on the
# validation split; the test split is evaluated once at the end.
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from .data import OASISDataset, scan_level_predictions
from .train_model_a import collate_fn, evaluate_test
from .train_model_b import build_model
from shared.reliability.fingerprint import FailureFingerprint


def find_failing_train_ids(model, train_csv: str, device: str, conf_threshold: float = 0.8) -> set[str]:
    per_scan = scan_level_predictions(model, OASISDataset(train_csv), device)
    bad = set()
    for sid, v in per_scan.items():
        p = v['probs']
        if int(p.argmax()) != v['label'] and float(p.max()) > conf_threshold:
            bad.add(str(sid))
    return bad


def val_loss_and_scan_acc(model, val_csv: str, device: str, crit) -> tuple[float, float]:
    loader = DataLoader(OASISDataset(val_csv), batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn)
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            tot += crit(model(imgs), labels).item() * len(labels)
            n += len(labels)
    per_scan = scan_level_predictions(model, OASISDataset(val_csv), device)
    acc = float(np.mean([int(v['probs'].argmax()) == v['label'] for v in per_scan.values()]))
    return tot / max(n, 1), acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--base-checkpoint', required=True)
    ap.add_argument('--fingerprint', default=None, help='Optional. Used only for category and notes, never to pick training cases.')
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-05)
    ap.add_argument('--oversample', type=float, default=5.0, help='Sampling weight multiplier for failing TRAIN subjects')
    ap.add_argument('--conf-threshold', type=float, default=0.8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_csv = os.path.join(args.data_root, 'splits/train.csv')
    val_csv = os.path.join(args.data_root, 'splits/val.csv')
    test_csv = os.path.join(args.data_root, 'splits/test.csv')
    model = build_model().to(device)
    state = torch.load(args.base_checkpoint, map_location=device)
    model.load_state_dict(state)
    fp_info = {}
    if args.fingerprint:
        fp = FailureFingerprint.from_json(args.fingerprint)
        test_ids = set(pd.read_csv(test_csv)['ID'].astype(str))
        fp_ids = {c.case_id for c in fp.evidence_cases}
        fp_info = {'fingerprint_id': fp.id, 'category': fp.failure_category,
                   'fingerprint_cases': len(fp_ids), 'fingerprint_cases_in_test_split': len(fp_ids & test_ids)}
        if fp_ids & test_ids:
            print(f'NOTE: {len(fp_ids & test_ids)} fingerprint cases are test cases; they are NOT used to select repair data.')
    failing = find_failing_train_ids(model, train_csv, device, args.conf_threshold)
    print(f'Confident errors of the base model on the train split: {len(failing)} scans')
    if not failing:
        res = {'repair': {**fp_info, 'status': 'not_run', 'reason': 'no confident train-split errors to oversample'}}
        with open(os.path.join(args.out, 'model_c_test_results.json'), 'w') as f:
            json.dump(res, f, indent=2)
        print(json.dumps(res, indent=2))
        return
    train_ds = OASISDataset(train_csv)
    weights = [args.oversample if str(r['ID']) in failing else 1.0 for (_, r) in train_ds.df.iterrows()]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=2, collate_fn=collate_fn)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    base_val = val_loss_and_scan_acc(model, val_csv, device, crit)
    best = {'val_loss': base_val[0], 'epoch': 0, 'state': {k: v.cpu().clone() for k, v in model.state_dict().items()}}
    print(f'epoch 0 (base): val_loss={base_val[0]:.4f} val_scan_acc={base_val[1]:.4f}', flush=True)
    for epoch in range(args.epochs):
        model.train()
        tot, correct, n = 0.0, 0, 0
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()
            tot += loss.item() * len(labels)
            correct += (out.argmax(1) == labels).sum().item()
            n += len(labels)
        vl, va = val_loss_and_scan_acc(model, val_csv, device, crit)
        print(f'epoch {epoch + 1}: train_loss={tot / max(n, 1):.4f} train_acc={correct / max(n, 1):.4f} val_loss={vl:.4f} val_scan_acc={va:.4f}', flush=True)
        if vl < best['val_loss']:
            best = {'val_loss': vl, 'epoch': epoch + 1, 'state': {k: v.cpu().clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best['state'])
    res = evaluate_test(model, test_csv, device)
    torch.save(model.state_dict(), os.path.join(args.out, 'model_c_vit_repaired.pt'))
    with open(os.path.join(args.out, 'model_c_test_results.json'), 'w') as f:
        json.dump({'repair': {**fp_info, 'status': 'run', 'selection': 'confident train-split errors of base model', 'oversample': args.oversample, 'lr': args.lr, 'epochs': args.epochs, 'best_epoch_by_val_loss': best['epoch'], 'failing_train_scans': len(failing)}, 'test': res}, f, indent=2)
    print(json.dumps({k: res[k] for k in ('accuracy', 'balanced_accuracy', 'roc_auc')}, indent=2))
    print("Compare against the base model's test results, once. If it is not better without hurting overall performance, report the repair as unsuccessful. That is a valid scientific finding.")


if __name__ == '__main__':
    main()
