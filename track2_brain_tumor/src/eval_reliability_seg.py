# Track 2 segmentation reliability + disagreement.
from __future__ import annotations
import argparse
import json
import os
import numpy as np
from shared.reliability.disagreement import segmentation_disagreement
from shared.reliability.gate import risk_coverage_curve

def region_calibration(prob_maps: np.ndarray, gt: np.ndarray, n_bins: int=10) -> dict:
    conf = prob_maps.max(axis=1)
    pred = prob_maps.argmax(axis=1)
    correct = (pred == gt).astype(float)
    rows = []
    for b in range(n_bins):
        (lo, hi) = (b / n_bins, (b + 1) / n_bins)
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            rows.append({'bin': f'({lo:.1f},{hi:.1f}]', 'count': int(m.sum()), 'mean_confidence': float(conf[m].mean()), 'empirical_accuracy': float(correct[m].mean())})
    return {'bins': rows, 'ece': float(sum((r['count'] / len(gt) * abs(r['mean_confidence'] - r['empirical_accuracy']) for r in rows)))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-a', required=True, help='Model A prediction dir (val split)')
    ap.add_argument('--pred-b', default=None, help='Model B prediction dir (optional)')
    ap.add_argument('--gt-dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import nibabel as nib
    report = {'per_case': []}
    for fn in sorted(os.listdir(args.pred_a)):
        if not fn.endswith('.nii.gz'):
            continue
        a = nib.load(os.path.join(args.pred_a, fn)).get_fdata().astype(int)
        g = nib.load(os.path.join(args.gt_dir, fn)).get_fdata().astype(int)
        g = np.where(g == 4, 3, g)
        case = {'case_id': fn.replace('.nii.gz', '')}
        if args.pred_b:
            b_path = os.path.join(args.pred_b, fn)
            if os.path.exists(b_path):
                b = nib.load(b_path).get_fdata().astype(int)
                case['disagreement'] = segmentation_disagreement(a, b)
        report['per_case'].append(case)
    with open(os.path.join(args.out, 'track2_reliability.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {len(report['per_case'])} case records. Voxel calibration requires saved probability maps (--save-probabilities on predict).")
if __name__ == '__main__':
    main()
