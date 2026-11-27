# Audit branch status (interim, read this first)

Branch `improve/reliability-audit` (draft PR) holds a reliability audit of the code and results. It is **interim**.

## What changed (code, tested on SYNTHETIC data only)
- `shared/reliability/gate.py`: gate objective and normaliser fitted on validation data only.
- `shared/reliability/ood.py`: optional Ledoit-Wolf shrinkage for the OOD covariance.
- `shared/reliability/uncertainty.py`: MC dropout counts only modules with p>0.
- Track 1 and Track 2 eval scripts use the above. HD95 is surfaced next to the original value (the original is kept; the two are not comparable).
- `tests/`: 15 synthetic tests. They check code behaviour, not model performance. Run: `python -m pytest tests`.

## What did NOT change
- No reported number was changed. The fixes have **not** been rerun on real data yet. Legacy results (including the old gate, which accepted everything) stay as published and are separate from these fixes.

## Results data added
- `docs/results/track2_flair_removal_per_case_56.csv`: all 56 FLAIR-removal cases, recovered from the executed notebook 05 output. Mean stability Dice 0.4176, 51 of 56 unstable (Dice <= 0.8).
- `docs/results/bootstrap_cis.csv`: CIs on independent units (scans or cases). Slice-level quantities get no CI because slices of one scan are correlated. Regenerate: `python scripts/bootstrap_cis.py`.

## Figure status (important)
`docs/figures/fig_flair_stability.pdf` on this branch is the OLD figure: it plots 49 cases under an n=56 label. The browser upload of the corrected PDF was blocked, so it is not on this branch yet. To regenerate the correct figure from the verified CSV:

    python scripts/make_flair_figure.py

It prints `n=56 mean=0.4176 unstable(<=0.8)=51` and overwrites the PDF. Until the corrected PDF is committed, treat the old figure as superseded by `track2_flair_removal_per_case_56.csv`.

## Not included
No scans, checkpoints or credentials. The OASIS terms and Kaggle dataset visibility are tracked outside this repo.
