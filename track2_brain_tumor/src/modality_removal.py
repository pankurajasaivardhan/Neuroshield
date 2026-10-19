# Track 2 counterfactual experiment: remove one modality and compare.
from __future__ import annotations
import argparse
import json
import os
import subprocess
import numpy as np
import nibabel as nib
import pandas as pd
MODALITY_CODES = {'flair': '0000', 't1': '0001', 't1ce': '0002', 't2': '0003'}

def stage_cases(case_ids, raw_data, out_dir, remove: str | None) -> str:
    images = os.path.join(out_dir, 'imagesTs')
    os.makedirs(images, exist_ok=True)
    for cid in case_ids:
        cdir = None
        for (root, dirs, _) in os.walk(raw_data):
            if cid in dirs:
                cdir = os.path.join(root, cid)
                break
        for (mod, code) in MODALITY_CODES.items():
            src = os.path.join(cdir, f'{cid}_{mod}.nii.gz')
            dst = os.path.join(images, f'{cid}_{code}.nii.gz')
            if mod == remove:
                img = nib.load(src)
                zeroed = nib.Nifti1Image(np.zeros_like(img.get_fdata()), img.affine, img.header)
                nib.save(zeroed, dst)
            elif not os.path.exists(dst):
                os.link(src, dst)
    return images

def run_predict(images, preds, artifact, config, fold):
    env = dict(os.environ)
    env['nnUNet_results'] = artifact
    subprocess.run(['nnUNetv2_predict', '-i', images, '-o', preds, '-d', 'Dataset001_BraTS2020', '-c', config, '-f', str(fold), '-tr', 'nnUNetTrainer_100epochs', '--disable_tta', '-npp', '1', '-nps', '1'], check=True, env=env)

def wt_dice(a: np.ndarray, b: np.ndarray) -> float:
    (a, b) = (a > 0, b > 0)
    inter = (a & b).sum()
    denom = a.sum() + b.sum()
    return float(2 * inter / denom) if denom else 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--raw-data', required=True)
    ap.add_argument('--artifact', required=True)
    ap.add_argument('--remove', default='flair', choices=list(MODALITY_CODES) + ['none'])
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default='2d')
    ap.add_argument('--fold', type=int, default=0)
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    case_ids = df[df['split'] == args.split]['case_id'].tolist()
    base_dir = os.path.join(args.out, 'full')
    imgs = stage_cases(case_ids, args.raw_data, base_dir, remove=None)
    run_predict(imgs, os.path.join(base_dir, 'predictions'), args.artifact, args.config, args.fold)
    rows = []
    if args.remove != 'none':
        int_dir = os.path.join(args.out, f'no_{args.remove}')
        imgs = stage_cases(case_ids, args.raw_data, int_dir, remove=args.remove)
        run_predict(imgs, os.path.join(int_dir, 'predictions'), args.artifact, args.config, args.fold)
        csv_path = os.path.join(args.out, 'modality_removal_per_case.csv')
        for cid in case_ids:
            full = nib.load(os.path.join(base_dir, 'predictions', f'{cid}.nii.gz')).get_fdata().astype(int)
            intr = nib.load(os.path.join(int_dir, 'predictions', f'{cid}.nii.gz')).get_fdata().astype(int)
            d = wt_dice(full, intr)
            rows.append({'case_id': cid, 'removed_modality': args.remove, 'wt_dice_full_vs_intervened': d, 'stable': bool(d > 0.8)})
            summ = {'intervention': f'{args.remove} removed (channel zeroed, documented)', 'split': args.split, 'n_cases': len(rows), 'mean_stability_dice': float(np.mean([r['wt_dice_full_vs_intervened'] for r in rows])), 'unstable_cases': [r['case_id'] for r in rows if not r['stable']], 'complete': len(rows) == len(case_ids), 'note': 'Low stability Dice means the prediction changes under modality removal - a counterfactual instability signal for the Reliability Gate.'}
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            with open(os.path.join(args.out, 'modality_removal_summary.json'), 'w') as f:
                json.dump(summ, f, indent=2)
            print(f"case {len(rows)}/{len(case_ids)} {cid}: wt_dice={d:.4f}", flush=True)
    else:
        summ = {'intervention': 'none (control)', 'n_cases': len(case_ids)}
    print(json.dumps(summ, indent=2))
if __name__ == '__main__':
    main()
