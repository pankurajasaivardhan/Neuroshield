# Track 1 Model A: ResNet-50 2.5D reference baseline (GPU: Kaggle/Colab).
from __future__ import annotations
import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from .data import OASISDataset, scan_level_predictions

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_model() -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    ids = [b[2] for b in batch]
    return (imgs, labels, ids)

def train_one_seed(seed: int, train_csv: str, val_csv: str, epochs: int, patience: int, device: str, model_builder=None) -> tuple[nn.Module, dict]:
    if model_builder is None:
        model_builder = build_model
    set_seed(seed)
    train_ds = OASISDataset(train_csv)
    val_ds = OASISDataset(val_csv)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2, collate_fn=collate_fn)
    model = model_builder().to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    crit = nn.CrossEntropyLoss()
    (best_val_loss, bad_epochs, best_state) = (float('inf'), 0, None)
    for epoch in range(epochs):
        model.train()
        for (imgs, labels, _) in train_loader:
            (imgs, labels) = (imgs.to(device), labels.to(device))
            opt.zero_grad()
            loss = crit(model(imgs), labels)
            loss.backward()
            opt.step()
        model.eval()
        (val_loss, n) = (0.0, 0)
        with torch.no_grad():
            for (imgs, labels, _) in val_loader:
                (imgs, labels) = (imgs.to(device), labels.to(device))
                val_loss += crit(model(imgs), labels).item() * len(labels)
                n += len(labels)
        val_loss /= max(n, 1)
        print(f'[seed {seed}] epoch {epoch + 1}: val_loss={val_loss:.4f}', flush=True)
        if val_loss < best_val_loss - 0.0001:
            (best_val_loss, bad_epochs) = (val_loss, 0)
            best_state = {k: v.cpu().clone() for (k, v) in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return (model, {'best_val_loss': best_val_loss})

def evaluate_test(model, test_csv: str, device: str) -> dict:
    test_ds = OASISDataset(test_csv)
    per_scan = scan_level_predictions(model, test_ds, device)
    y_true = np.array([v['label'] for v in per_scan.values()])
    probs = np.stack([v['probs'] for v in per_scan.values()])
    y_pred = probs.argmax(axis=1)
    return {'accuracy': float(accuracy_score(y_true, y_pred)), 'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)), 'sensitivity_recall': float(recall_score(y_true, y_pred)), 'specificity': float(recall_score(y_true, y_pred, pos_label=0)), 'f1': float(f1_score(y_true, y_pred)), 'roc_auc': float(roc_auc_score(y_true, probs[:, 1])), 'n_scans': int(len(y_true)), 'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(), 'test_probs': probs.tolist(), 'test_labels': y_true.tolist()}

def run(model_builder=None, model_name: str='model_a_resnet50'):
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 2024])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_csv = os.path.join(args.data_root, 'splits/train.csv')
    val_csv = os.path.join(args.data_root, 'splits/val.csv')
    test_csv = os.path.join(args.data_root, 'splits/test.csv')
    all_results = {}
    for seed in args.seeds:
        (model, info) = train_one_seed(seed, train_csv, val_csv, args.epochs, args.patience, device, model_builder=model_builder)
        res = evaluate_test(model, test_csv, device)
        res.update(info)
        all_results[seed] = res
        torch.save(model.state_dict(), os.path.join(args.out, f'{model_name}_seed{seed}_best.pt'))
        with open(os.path.join(args.out, f'{model_name}_results_seed{seed}.json'), 'w') as f:
            json.dump(res, f, indent=2)
        print(f"[seed {seed}] test accuracy={res['accuracy']:.4f}", flush=True)
    keys = ['accuracy', 'balanced_accuracy', 'sensitivity_recall', 'specificity', 'f1', 'roc_auc']
    summary = {k: {'mean': float(np.mean([r[k] for r in all_results.values()])), 'std': float(np.std([r[k] for r in all_results.values()]))} for k in keys}
    with open(os.path.join(args.out, f'{model_name}_results_summary.json'), 'w') as f:
        json.dump({'per_seed': all_results, 'summary': summary, 'config': {'model': model_name, 'pretrained': 'IMAGENET1K_V1', 'seeds': args.seeds, 'epochs': args.epochs, 'patience': args.patience}}, f, indent=2)
    print('Summary:', json.dumps(summary, indent=2))

def main():
    run()
if __name__ == '__main__':
    main()
