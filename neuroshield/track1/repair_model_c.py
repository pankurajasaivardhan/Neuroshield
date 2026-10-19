# Track 1 Model C: targeted repair driven by a Failure Fingerprint.
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from .data import OASISDataset
from .train_model_a import collate_fn, evaluate_test
from .train_model_b import build_model
from neuroshield.shared.reliability.fingerprint import FailureFingerprint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--base-checkpoint', required=True)
    ap.add_argument('--fingerprint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-05)
    ap.add_argument('--oversample', type=float, default=5.0, help='Sampling weight multiplier for failing subjects')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp = FailureFingerprint.from_json(args.fingerprint)
    failing_ids = {c.case_id for c in fp.evidence_cases}
    train_csv = os.path.join(args.data_root, 'splits/train.csv')
    val_csv = os.path.join(args.data_root, 'splits/val.csv')
    test_csv = os.path.join(args.data_root, 'splits/test.csv')
    train_ids = set(pd.read_csv(train_csv)['ID'].astype(str))
    usable = failing_ids & train_ids
    print(f'Fingerprint cases: {len(failing_ids)}, usable in train split: {len(usable)}')
    train_ds = OASISDataset(train_csv)
    weights = [args.oversample if str(r['ID']) in usable else 1.0 for (_, r) in train_ds.df.iterrows()]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    loader = DataLoader(train_ds, batch_size=16, sampler=sampler, num_workers=2, collate_fn=collate_fn)
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.base_checkpoint, map_location=device))
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    crit = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train()
        (tot, correct, n) = (0.0, 0, 0)
        for (imgs, labels, _) in loader:
            (imgs, labels) = (imgs.to(device), labels.to(device))
            opt.zero_grad()
            loss = crit(model(imgs), labels)
            loss.backward()
            opt.step()
            tot += loss.item() * len(labels)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            n += len(labels)
        print(f'epoch {epoch + 1}: train_loss={tot / max(n, 1):.4f} train_acc={correct / max(n, 1):.4f}', flush=True)
    res = evaluate_test(model, test_csv, device)
    torch.save(model.state_dict(), os.path.join(args.out, 'model_c_vit_repaired.pt'))
    with open(os.path.join(args.out, 'model_c_test_results.json'), 'w') as f:
        json.dump({'repair': {'fingerprint_id': fp.id, 'category': fp.failure_category, 'oversample': args.oversample, 'lr': args.lr, 'epochs': args.epochs, 'usable_train_cases': len(usable)}, 'test': res}, f, indent=2)
    print(json.dumps({k: res[k] for k in ('accuracy', 'balanced_accuracy', 'roc_auc')}, indent=2))
    print("Compare against the base model's test results. If not better on the failing category without hurting overall performance, report the repair as unsuccessful. That is a valid scientific finding.")
if __name__ == '__main__':
    main()
