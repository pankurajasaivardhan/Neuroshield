# Version 5 run (Kaggle, private notebook, 549.7 s)

Descriptive reanalysis of an already-seen test split. Seed 42 checkpoints, no retraining, 36 test scans, 35 validation scans.

- `v5_scan_level_results.json`: provenance (commit, gate.py path, gate modes, checkpoint SHA-256) and the full scan-level output printed by the run.
- `v5_scan_level_test_k_of_n.csv`: errors among accepted scans / accepted scans per coverage point and for the validation-chosen operating points.

Notes
- `test_aurc` in the JSON is a PARTIAL area: trapezoid rule over coverage 0.5 to 1.0 only (six points, width 0.5). It is not the standard AURC over 0 to 1.
- Bootstrap intervals (10,000 resamples, seed 0) include zero everywhere. That does not show equivalence or rule out a benefit. No ranking is claimed.
- ResNet-50 seed 42: 25/36 scans correct (matches the training notebook). ViT-B/16 seed 42: 22/36 here versus 26/36 in the training notebook. Cause not found. The checkpoint SHA-256 matches the repository LFS pointer.
- Slice-level v2 gate (cutoff 0.521, validation coverage 0.58) is a different quantity from the scan protocol grid (coverage >= 0.6).
