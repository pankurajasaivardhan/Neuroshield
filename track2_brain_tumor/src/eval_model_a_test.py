# Track 2 held-out test evaluation: Dice, IoU, HD95 on the 56 cases.
from __future__ import annotations
import argparse
import json
import os
import subprocess
from scipy.ndimage import distance_transform_edt
import numpy as np
import nibabel as nib
REGIONS = {'WT': lambda m: m > 0, 'TC': lambda m: (m == 1) | (m == 3), 'ET': lambda m: m == 3}

def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    inter = float((pred & gt).sum())
    (p, g) = (float(pred.sum()), float(gt.sum()))
    if p + g == 0:
        return (1.0, 1.0)
    dice = 2 * inter / (p + g)
    union = float((pred | gt).sum())
    iou = inter / union if union > 0 else 0.0
    return (dice, iou)

def hd95(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float('nan')
    dt_gt = distance_transform_edt(~gt.astype(bool), sampling=spacing)
    dt_pred = distance_transform_edt(~pred.astype(bool), sampling=spacing)
    d1 = dt_gt[pred.astype(bool)]
    d2 = dt_pred[gt.astype(bool)]
    return float(np.percentile(np.concatenate([d1, d2]), 95))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, help='braintumor_manifest.csv')
    ap.add_argument('--raw-data', required=True, help='BraTS2020_TrainingData root')
    ap.add_argument('--artifact', required=True, help='Recovered model dir (parent of Dataset001_BraTS2020)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default='2d')
    ap.add_argument('--fold', type=int, default=0)
    args = ap.parse_args()
    import pandas as pd
    df = pd.read_csv(args.manifest)
    test_ids = df[df['split'] == 'test']['case_id'].tolist()
    assert len(test_ids) == 56, f'expected 56 test cases, got {len(test_ids)}'
    work = os.path.join(args.out, 'test_work')
    images_ts = os.path.join(work, 'imagesTs')
    preds_dir = os.path.join(work, 'predictions')
    os.makedirs(images_ts, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)
    modality_files = {'flair': '0000', 't1': '0001', 't1ce': '0002', 't2': '0003'}
    case_dirs = {}
    for (root, dirs, _) in os.walk(args.raw_data):
        for d in dirs:
            if d.startswith('BraTS20_Training_'):
                case_dirs[d] = os.path.join(root, d)
    for cid in test_ids:
        cdir = case_dirs[cid]
        for (mod, code) in modality_files.items():
            src = os.path.join(cdir, f'{cid}_{mod}.nii.gz')
            dst = os.path.join(images_ts, f'{cid}_{code}.nii.gz')
            if not os.path.exists(dst):
                os.link(src, dst) if os.path.exists(src) else None
    env = dict(os.environ)
    env['nnUNet_results'] = args.artifact
    subprocess.run(['nnUNetv2_predict', '-i', images_ts, '-o', preds_dir, '-d', 'Dataset001_BraTS2020', '-c', args.config, '-f', str(args.fold), '-tr', 'nnUNetTrainer_100epochs', '--disable_tta'], check=True, env=env)
    rows = []
    for cid in test_ids:
        pred = nib.load(os.path.join(preds_dir, f'{cid}.nii.gz')).get_fdata().astype(int)
        gt = nib.load(os.path.join(case_dirs[cid], f'{cid}_seg.nii.gz')).get_fdata().astype(int)
        gt = np.where(gt == 4, 3, gt)
        spacing = nib.load(os.path.join(case_dirs[cid], f'{cid}_seg.nii.gz')).header.get_zooms()[:3]
        row = {'case_id': cid}
        for (region, fn) in REGIONS.items():
            (p_m, g_m) = (fn(pred), fn(gt))
            (d, i) = dice_iou(p_m, g_m)
            row[f'{region}_dice'] = d
            row[f'{region}_iou'] = i
            row[f'{region}_hd95_mm'] = hd95(p_m, g_m, spacing)
        rows.append(row)
        print(cid, {k: round(v, 4) if isinstance(v, float) else v for (k, v) in row.items() if k != 'case_id'}, flush=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(os.path.join(args.out, 'model_a_test_per_case.csv'), index=False)
    agg = out_df.drop(columns=['case_id']).mean(numeric_only=True).to_dict()
    with open(os.path.join(args.out, 'model_a_test_summary.json'), 'w') as f:
        json.dump({'n_cases': len(rows), 'config': args.config, 'fold': args.fold, 'aggregate_mean': agg, 'note': 'Held-out 56-case outer test set. These are the final Model A test metrics; the earlier 0.7651 was internal validation.'}, f, indent=2)
    print(json.dumps(agg, indent=2))
if __name__ == '__main__':
    main()
