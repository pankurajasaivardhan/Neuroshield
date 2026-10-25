# NeuroShield demo website

Web demo for the NeuroShield research project: upload a brain MRI and watch
the real trained models run, with reliability signals and abstention.

- Track 1 - Alzheimer's classification: 3-seed ViT-B/16 ensemble (OASIS-1)
- Track 2 - Brain tumor segmentation: 2D nnU-Net (BraTS 2020), with an optional
  FLAIR-removal reliability stress test

One interface, two separate models. Research demo, not a diagnostic device.

## Layout

- `server/` - FastAPI backend (`app.py`), model runners (`runners/`), static frontend (`static/`)
- `deploy/local/` - pinned CPU dependencies (`requirements.txt` for Linux,
  `requirements-mac.txt` for macOS Apple Silicon, both wheel-checked)
- `run_mac.command` - double-click start on macOS (see SETUP_MAC.md)
- `run_local.sh` - start script for Linux

Both run scripts download the trained weights automatically from the
repository's GitHub Release (weights only, no scan data), install CPU
dependencies, and serve the app on port 8000 - real uploads, real inference.
Needs about 4 GB free RAM.

Tested end-to-end on Linux (Kaggle, CPU). On macOS Apple Silicon the pinned
dependency wheels are verified available for Python 3.12; the first full
Mac run is pending.

## Data, licensing and attribution

- No MRI scans are bundled in this repository. The OASIS-1 Data Use Terms do
  not allow redistributing the data to third parties without written
  permission, and the BraTS data usage terms do not grant redistribution
  rights. Upload your own compatible scans.
- Alzheimer's models trained on OASIS-1: Cross-Sectional (Marcus et al.,
  2007, doi:10.1162/jocn.2007.19.9.1498). Tumor model trained on BraTS 2020
  (Menze et al. 2015; Bakas et al. 2017, 2018).
