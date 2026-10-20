# Run history and transparency notes

Every number in this repository comes from a real, logged Kaggle run. This file
records what was run, with what settings, what failed and was re-run, and every
place where the final state differs from the original plan. Nothing is
cherry-picked: negative and degenerate results are reported as found.

## What was actually run (final state, 2026-10-21)

All runs were executed on Kaggle under two accounts (Track 1: `mohanroyal273`,
Track 2: `rajasaivardhan`). The executed notebooks with full cell outputs are
in `track1_alzheimers/notebooks/` and `track2_brain_tumor/notebooks/`.

### Track 1 - Alzheimer's classification (OASIS-1)

| Run | Notebook version | Result |
|---|---|---|
| Model A training (ResNet-50, lr=1e-5, 3 seeds) | `track1-model-a-resnet50-baseline` v4 | scan-level acc 0.7130 +/- 0.0131, balanced 0.7063 +/- 0.0184, sens 0.6667 +/- 0.0544, spec 0.7460 +/- 0.0224, F1 0.6583 +/- 0.0279, AUC 0.7270 +/- 0.0069 (36 test scans) |
| Model B training (ViT-B/16, 3 seeds) | `03-train-model-b-final` v1 (executed 2026-09-24) | scan acc 0.7130 +/- 0.0131, AUC 0.7386 +/- 0.0120; majority baseline 0.5732 |
| Reliability evaluation | `notebooke0cf27f81b` v3 | slice-level, 180 test slices - see table below |

Track 1 reliability (slice level, 180 test slices):

| Signal | Model A (ResNet-50) | Model B (ViT) |
|---|---|---|
| ECE raw | 0.0915 | 0.1238 |
| ECE calibrated | 0.0829 | 0.0672 |
| Brier | 0.4358 | 0.4706 |
| Accuracy | 0.6111 | 0.5722 |
| Mean OOD score | 8.49e6 (ill-conditioned, see notes) | 8252.9 |

Disagreement: mismatch fraction 0.3944, mean JS divergence 0.0273 (180 slices).
Gate: DEGENERATE - validation-fitted thresholds admit every input (cutoff 1.00),
so accepted error equals ungated test error 0.3889. The threshold-swept
risk-coverage curve is NOT flat (accepted-subset error ~0.18-0.20 at 15-35%
coverage), i.e. the reliability score's ranking carries signal that the fitted
thresholds cannot exploit at this cohort size. Failure fingerprint
`fp-25b871fd`: overconfident misclassification (errors at confidence > 0.8).

### Track 2 - Brain tumor segmentation (BraTS 2020)

Split (fold0, authoritative): 249 train / 63 val / 56 test, from
`splits_final.json` under `nnUNet_preprocessed`.

| Run | Notebook version | Result |
|---|---|---|
| Model A training (2D nnU-Net) | `braintumor-01-final` (latest committed) | see test metrics below |
| Model A held-out test | `track2-model-a-test-metrics` v5 | 56 cases: foreground Dice 0.9005 +/- 0.0787; per-label Dice l1 0.6519 +/- 0.2970, l2 0.7769 +/- 0.1696, l3 0.7804 +/- 0.2129 |
| Model B training (UNETR, MONAI, 96^3 patches, DiceCE, 40 epochs) | `track2-model-b-unetr` v5 | best val mean Dice 0.6305 (epoch 40); curve 0.5765 / 0.6069 / 0.6159 / 0.6305 at epochs 10/20/30/40 |
| FLAIR-removal counterfactual (Model A) | `track2-flair-removal` v6 | 56/56 cases: mean stability Dice 0.4176; 51/56 unstable (WT Dice <= 0.8) |
| Model B eval + reliability | `notebookc85717a0d1` v3 | UNETR test (n=56): WT Dice 0.7913, TC 0.6279, ET 0.6358 (HD95 8.86 / 13.32 / 12.73 mm); val (n=63): WT 0.8188, TC 0.7104, ET 0.6803. A-vs-B agreement (test): mean foreground disagreement 0.3688; per-class IoU necrotic 0.3578 / edema 0.5938 / enhancing 0.5750 |

## Deviations from the original plan (honest notes)

1. **UNETR trained for 40 epochs, not 200.** The training script's original
   default was 200 epochs. The 40-epoch run was kept as the final Model B
   (validation Dice was still improving slowly at epoch 40: 0.6159 -> 0.6305).
   Repo defaults were updated to match what was actually run.
2. **A 12.9-hour UNETR training run was lost.** The likely cause is Kaggle's
   ~12-hour GPU session wall (inferred, not confirmed; the session died and its
   outputs were not persisted). The final v5 run (9h48m) completed within the
   window.
3. **Split string correction 257/55 -> 249/63/56.** Older notebook markdown
   (including the executed `braintumor-01-final` STAGE-3 heading) still says
   257/55/56. That string is stale: the split actually used by every reported
   result is fold0 = 249/63/56 from `splits_final.json`. The executed notebooks
   are kept as they really ran; the correction is documented here and in the
   README rather than edited into history.
4. **Track 2 reliability ran on the test split, not validation.** Model A
   validation predictions were never persisted (the `neuroshield-track2-model-a`
   dataset contains only `test_predictions/`). The segmentation reliability
   script fits no thresholds, so running it on the 56-case test set introduces
   no supervision leak. Eval attempts v1 (failed fast on the missing validation
   predictions) and v2 (failed on a CUDA-tensor `.numpy()` staging bug) preceded
   the successful v3; both failures were diagnosed and fixed in the open.
5. **Track 1 gate degenerates (negative result).** Validation-fitted accept
   thresholds admit all 180 test slices. Reported as found; no gate is claimed.
6. **Track 1 Model A OOD signal is ill-conditioned.** Mahalanobis OOD scores
   (~8.5e6) reflect a degenerate feature covariance on this cohort, not a tuned
   detector. Reported as an honest negative measurement.
7. **Model C repair was not triggered in either track.** Track 1's audit found
   overconfident misclassification without a repairable subgroup; retraining to
   chase calibration on 36 test scans would fit noise. Recorded as a valid
   negative repair outcome per the project rules.
8. **ADNI external evaluation was not run** (GPU quota). Documented as a
   limitation, not a result.
9. **2D vs 3D nnU-Net for Model A** remains an open decision; no retrain was
   done. Voxel-level calibration for segmentation was not possible because
   probability maps were not saved during prediction (`--save-probabilities`).
10. **Track 1 reliability metrics are slice-level** (180 test slices), while
    the headline classification metrics are scan-level (36 scans, 3 seeds).
    They answer different questions and are labeled as such wherever reported.

## Figures

Vector PDFs in `docs/figures/`, generated from the raw per-run data in
`docs/results/`:

- `fig_unetr_val_curve.pdf` - UNETR validation Dice by epoch
- `fig_flair_stability.pdf` - FLAIR-removal stability distribution
- `fig_track1_risk_coverage.pdf` - Track 1 risk-coverage sweep
- `fig_track2_agreement_iou.pdf` - Track 2 per-case agreement IoU by class
- `fig_track1_js_divergence.pdf` - Track 1 JS-divergence distribution
