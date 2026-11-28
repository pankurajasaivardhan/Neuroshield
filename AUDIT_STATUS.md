# Audit status (interim, read this first)

This file records what the reliability audit changed, what was run, and what is still open. Reported results are interim.

## Read this before using Track 1 reliability numbers
- **ViT input mismatch.** The ViT-B/16 training notebook builds its input from 3 adjacent sagittal slices stacked as channels, with percentile normalisation over the stack, and averages raw softmax over slices. The repository evaluation code (`OASISDataset`) repeats one slice across 3 channels. So the ViT reliability, calibration, gate, OOD and disagreement numbers (including the 39.4% slice disagreement and ViT ECE) do not apply to the notebook-trained ViT. This comes from comparing code. It is not yet confirmed by a run.
- **Evidence for the mismatch.** In Version 5 the ViT checkpoint (SHA-256 matches the repository LFS pointer) classified 22 of 36 scans correctly, against 26 of 36 in its training notebook. The ResNet-50 gave 25 of 36 in both.
- **Fix and rerun.** `--vit-input notebook` in `track1_alzheimers/src/eval_reliability.py` reproduces the notebook input (`OASISDatasetNotebookViT`). Per-scan logits are saved so the raw-probability averaging check can be done. The matched-input run (Kaggle Version 6, private notebook) was submitted and is **pending**. No number from it is reported here.
- **ResNet-50.** Its training notebook uses the repository `train_model_a`, which uses the repository `OASISDataset`, the same input as the evaluation. The exact repository commit that notebook cloned was not confirmed.

## Repaired gate: executed, no demonstrated advantage
- The original gate admits everything: its target is the median of binary validation errors, which is 0 when the error rate is under 50%. It is kept as legacy and tagged as such.
- The repaired gate (`shared/reliability/gate.py`, validation-fitted) ran on the real splits in Version 5 (slice cutoff 0.521, validation coverage 0.58) and on synthetic tests.
- At scan level (36 test scans, seed 42), compared with a maximum-probability baseline, every paired bootstrap interval includes 0. That shows neither equivalence nor benefit. No ranking is claimed. Details: `docs/results/V5_README.md`, `docs/results/v5_scan_level_results.json`, `docs/results/v5_scan_level_test_k_of_n.csv`. The ViT rows are not valid for the notebook-trained ViT (see above).

## What changed in code (tested on SYNTHETIC data)
- `shared/reliability/gate.py`, `ood.py`, `uncertainty.py`, `selective.py`, `scan_eval.py`; Track 1 and Track 2 eval scripts; `tests/` (synthetic tests of code behaviour, not model performance; run `python -m pytest tests`).
- HD95 is shown next to the original value. The original is kept and the two are not comparable.

## Results data added
- `docs/results/track2_flair_removal_per_case_56.csv`: all 56 FLAIR-removal cases (mean stability Dice 0.4176, 51 of 56 unstable). 24 of the 56 match the printed output of the committed modality-removal notebook, and case 004 was reproduced in a separate run. The other 32 rest on the committed file only.
- `docs/results/bootstrap_cis.csv`: intervals on independent units (scans or cases).
- UNETR validation and test tables were checked against the Kaggle outputs.

## Stale figure
`docs/figures/fig_flair_stability.pdf` is the OLD figure: it plots 49 cases under an n=56 label. The browser upload of the corrected PDF was rejected, so it is not in the repository. Regenerate it from the 56-case CSV with `python scripts/make_flair_figure.py` (prints `n=56 mean=0.4176 unstable(<=0.8)=51`).

## Not included
No scans, checkpoints or credentials. The OASIS terms and Kaggle dataset visibility are tracked outside this repository and are not fully resolved.
