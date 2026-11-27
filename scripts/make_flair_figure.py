"""Regenerate docs/figures/fig_flair_stability.pdf from the verified per-case CSV.

Source data: docs/results/track2_flair_removal_per_case_56.csv (all 56 test cases,
recovered from the committed notebook 05 output). Run from the repo root:

    python scripts/make_flair_figure.py
"""
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs')
with open(os.path.join(R, 'results', 'track2_flair_removal_per_case_56.csv'), newline='') as f:
    vals = [float(r['wt_dice_full_vs_intervened_4dp']) for r in csv.DictReader(f)]
assert len(vals) == 56, len(vals)
mean = sum(vals) / len(vals)
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.hist(vals, bins=[i / 10 for i in range(11)], color='#4c72b0', edgecolor='white')
ax.axvline(mean, color='black', linestyle='--', label=f'mean {mean:.4f}')
ax.axvline(0.8, color='#c44e52', linestyle=':', label='unstable threshold 0.8')
ax.set_xlabel('Stability Dice after FLAIR removal')
ax.set_ylabel(f'Cases (n={len(vals)})')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(R, 'figures', 'fig_flair_stability.pdf'))
print(f'n={len(vals)} mean={mean:.4f} unstable(<=0.8)={sum(v <= 0.8 for v in vals)}')
