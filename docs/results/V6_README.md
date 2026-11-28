# Version 6: matched-input ViT rerun (Track 1, seed 42)

Private Kaggle notebook, Version 6, 526 s, GPU. Repository commit b5f9f68e7872eab7b5e737d9debede7c682cd8d0, command with `--vit-input notebook`. Checkpoint SHA-256 values are the same as Version 5 (ResNet-50 505027c8...4310, ViT-B/16 30e76d9a...7de8). Descriptive reanalysis of an already-seen test split: one seed, 36 test scans, 180 test slices. Real run, not synthetic.

## What it shows
- ViT-B/16 scan accuracy: 26/36 (0.722) with the notebook input, 22/36 with the repository input (Version 5). The training notebook reported 26/36. This confirms the input mismatch. It does not by itself prove that every pipeline step matches the training notebook.
- ResNet-50: 25/36 in both versions. Its input is the repository input; the commit its training notebook cloned is unconfirmed.

## ViT slice level, test (180 slices)
| Quantity | Repo input (V5) | Matched (V6) |
|---|---|---|
| Accuracy | 0.5722 | 0.6389 |
| ECE raw | 0.1238 | 0.0730 |
| ECE temperature-scaled | 0.0672 | 0.0992 |
| Brier | 0.4706 | 0.4510 |
| Mean OOD (Mahalanobis) | 8252.9 | 7866.2 |

## Scan-level gate vs max-probability baseline (errors/accepted at coverage 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
- ViT gate: 5/18, 5/22, 6/26, 8/29, 10/33, 10/36. ViT confidence: 5/18, 6/22, 7/26, 7/29, 8/33, 10/36.
- Partial AURC (0.5 to 1.0): gate 0.1315, confidence 0.1304. Paired bootstrap difference 95% CI -0.024 to 0.026.
- ResNet-50 gate 0.1503, confidence 0.1438, difference CI -0.032 to 0.037 (unchanged from V5).
- All intervals include 0. No ranking or advantage is claimed, and the intervals do not show equivalence.
- Raw-probability averaging sensitivity: same accepted-error counts for both models; ViT difference CI -0.026 to 0.023.

## Model disagreement with the matched ViT input (test, 180 slices)
- Label mismatch fraction 0.2611 (47/180), against 0.394 (71/180) with the repository input. Mean JS divergence 0.0187 (median 0.0078; 20 slices above 0.05, 4 above 0.10, max 0.157), against 0.0273 before. Per-slice values: `v6_disagreement_matched.json`. They were read from `reliability_results.json` in the private Kaggle run output page (the npz files were not downloaded).

## Not recomputed
- The original-run ViT gate, OOD and risk-coverage sweep (Fig. 9 of the paper) remain legacy.
- The per-model logits (`logits_model_{a,b}_{val,test}.npz`) are in the private Kaggle run output; they are not in this repository.

Files: `v6_scan_level_results.json`, `v6_scan_level_test_k_of_n.csv`, `v6_run_log.txt` (this folder).
