# Track 2 Model B: UNETR training via MONAI (GPU: Kaggle).
from __future__ import annotations
import argparse
import json
import os
import numpy as np
import torch
import pandas as pd
from monai.networks.nets import UNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, EnsureTyped, AsDiscrete
from monai.data import Dataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
MODS = ['flair', 't1', 't1ce', 't2']

def case_dicts(manifest: str, raw_data: str, split: str) -> list[dict]:
    df = pd.read_csv(manifest)
    df = df[df['split'] == split]
    cases = []
    for (_, r) in df.iterrows():
        cid = r['case_id']
        for (root, dirs, _) in os.walk(raw_data):
            if cid in dirs:
                cdir = os.path.join(root, cid)
                cases.append({'image': [os.path.join(cdir, f'{cid}_{m}.nii') for m in MODS], 'label': os.path.join(cdir, f'{cid}_seg.nii'), 'case_id': cid})
                break
    return cases

def transforms_train():
    return Compose([LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys=['image', 'label']), Orientationd(keys=['image', 'label'], axcodes='RAS'), Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')), ScaleIntensityRanged(keys=['image'], a_min=0, a_max=2000, b_min=0, b_max=1, clip=True), CropForegroundd(keys=['image', 'label'], source_key='image'), RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=2), RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0), RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1), RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2), RandRotate90d(keys=['image', 'label'], prob=0.5, max_k=3), EnsureTyped(keys=['image', 'label'])])

def transforms_val():
    return Compose([LoadImaged(keys=['image', 'label']), EnsureChannelFirstd(keys=['image', 'label']), Orientationd(keys=['image', 'label'], axcodes='RAS'), Spacingd(keys=['image', 'label'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')), ScaleIntensityRanged(keys=['image'], a_min=0, a_max=2000, b_min=0, b_max=1, clip=True), CropForegroundd(keys=['image', 'label'], source_key='image'), EnsureTyped(keys=['image', 'label'])])

def remap_label(batch_label):
    return torch.where(batch_label == 4, 3, batch_label)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--raw-data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--val-interval', type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = Dataset(case_dicts(args.manifest, args.raw_data, 'train'), transform=transforms_train())
    val_ds = Dataset(case_dicts(args.manifest, args.raw_data, 'val'), transform=transforms_val())
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    model = UNETR(in_channels=4, out_channels=4, img_size=(96, 96, 96), feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, norm_name='instance', res_block=True, dropout_rate=0.1).to(device)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-05)
    dice_metric = DiceMetric(include_background=False, reduction='mean')
    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)
    (best_dice, history) = (-1.0, [])
    for epoch in range(args.epochs):
        model.train()
        (epoch_loss, steps) = (0.0, 0)
        for batch in train_loader:
            x = batch['image'].to(device)
            y = remap_label(batch['label'].long()).to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            steps += 1
        epoch_loss /= max(steps, 1)
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            dice_metric.reset()
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['image'].to(device)
                    y = remap_label(batch['label'].long()).to(device)
                    out = sliding_window_inference(x, (96, 96, 96), 4, model)
                    for (p, l) in zip(decollate_batch(out), decollate_batch(y)):
                        dice_metric(y_pred=[post_pred(p)], y=[post_label(l)])
            md = dice_metric.aggregate().item()
            print(f'epoch {epoch + 1}: loss={epoch_loss:.4f} val_dice={md:.4f}', flush=True)
            history.append({'epoch': epoch + 1, 'train_loss': epoch_loss, 'val_dice': md})
            if md > best_dice:
                best_dice = md
                torch.save(model.state_dict(), os.path.join(args.out, 'unetr_best.pt'))
                print(f'  new best val Dice {md:.4f}, checkpoint saved', flush=True)
    with open(os.path.join(args.out, 'unetr_history.json'), 'w') as f:
        json.dump({'best_val_dice': best_dice, 'history': history, 'config': {'arch': 'UNETR', 'img_size': [96, 96, 96], 'epochs': args.epochs, 'loss': 'DiceCE', 'split': 'same manifest as Model A (249/63/56)'}}, f, indent=2)
    print(f'Done. Best val Dice {best_dice:.4f}. Next: held-out test eval with eval_model_a_test.py adapted for UNETR, then reliability comparison.')
if __name__ == '__main__':
    main()
