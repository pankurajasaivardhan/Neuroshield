"""Percentile bootstrap CIs (10,000 resamples, seed 0) computed ONLY from the per-case files
committed in docs/results. Case-level resampling. Writes docs/results/bootstrap_cis.csv.
Track 1 slice rows are not independent (5 slices per scan, no scan id in the CSV), so no slice-level interval is written."""
import csv, os, numpy as np
R = os.path.join(os.path.dirname(__file__), '..', 'docs', 'results')
rng = np.random.default_rng(0)

def rows(name):
    with open(os.path.join(R, name)) as f:
        return list(csv.DictReader(f))

def boot(x, stat=np.mean, n=10000):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    s = np.array([stat(x[i]) for i in idx])
    return float(stat(x)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)), len(x)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - m) / d, (c + m) / d, n

out = []
rel = rows('track2_reliability_per_case.csv')
for col, label in [('foreground_disagreement_rate', 'T2 A-vs-B foreground disagreement rate (56 test cases)'),
                   ('iou_necrotic', 'T2 A-vs-B agreement IoU necrotic (test)'),
                   ('iou_edema', 'T2 A-vs-B agreement IoU edema (test)'),
                   ('iou_enhancing', 'T2 A-vs-B agreement IoU enhancing (test)')]:
    out.append((label, *boot([float(r[col]) for r in rel if r[col] not in ('', 'None')]), 'case-level bootstrap'))
fl = rows('track2_flair_removal_per_case_56.csv')  # all 56 cases, recovered from the committed notebook-05 output (4 dp)
st = [float(r['wt_dice_full_vs_intervened_4dp']) for r in fl]
out.append(('T2 FLAIR-removal stability Dice (all 56 cases)', *boot(st), 'case-level bootstrap; matches run-summary mean 0.4176'))
k = sum(r['stable'] == 'False' for r in fl)
out.append(('T2 unstable fraction after FLAIR removal (56 cases)', *wilson(k, len(fl)), 'Wilson 95%'))
out.append(('T2 unstable fraction after FLAIR removal (51/56 from run summary)', *wilson(51, 56), 'Wilson 95%'))
# Track 1 slice-level quantities (JS divergence, 71/180 mismatch) are deliberately NOT given an interval:
# track1_js_divergence.csv has no scan id, so slices of one scan cannot be grouped and a slice-level
# bootstrap would be too narrow. Scan-level Track 1 accuracy uses Wilson on 26/36 per model in the paper (ResNet seed 42 is 25/36).
out.append(('T1 single-model accuracy 26/36 scans (ViT seeds 42,123; ResNet seeds 123,2024)', *wilson(26, 36), 'Wilson 95%, scans are independent subjects, one model'))
with open(os.path.join(R, 'bootstrap_cis.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['quantity', 'estimate', 'ci95_low', 'ci95_high', 'n', 'method'])
    for r in out:
        w.writerow([r[0]] + [f'{v:.4f}' if isinstance(v, float) else v for v in r[1:5]] + [r[5]])
for r in out:
    print(f'{r[0]}: {r[1]:.4f} [{r[2]:.4f}, {r[3]:.4f}] n={r[4]}')
