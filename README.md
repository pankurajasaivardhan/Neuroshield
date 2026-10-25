# NeuroShield

A two-track medical AI reliability project on brain MRI. NeuroShield does
not only train models; it checks when a prediction can be trusted, records
why a model fails, repairs the specific weakness, and abstains with an
evidence-grounded explanation when a prediction is unreliable.

- Track 1 - Alzheimer's disease classification on OASIS structural MRI
- Track 2 - Brain tumor (glioma) segmentation on BraTS2020

Research prototype. Not a clinically validated diagnostic system. Not for
medical diagnosis.

## Repository structure

```
Neuroshield/
├── demo/                       # Self-contained demo website (run this)
│   ├── server/                   # FastAPI backend + model runners + frontend
│   ├── deploy/local/             # pinned CPU requirements (Linux + macOS)
│   ├── run_mac.command           # double-click start for macOS
│   ├── run_local.sh              # start script for Linux
│   └── SETUP_MAC.md              # step-by-step Mac guide
│
├── shared/                     # Reliability framework used by both tracks
│   ├── reliability/              # uncertainty, calibration, OOD, disagreement,
│   │                             # gate, failure fingerprint, report schema
│   ├── explain/                  # RAG + LLM explanation after abstention
│   └── app/                      # FastAPI backend + Streamlit frontend
│
├── track1_alzheimers/          # Track 1: Alzheimer's classification (OASIS)
│   ├── src/                      # reusable Python modules
│   │   ├── data.py                 # 2.5D OASIS dataset
│   │   ├── train_model_a.py        # ResNet-50 reference baseline
│   │   ├── train_model_b.py        # ViT-B/16 (finalized Model B)
│   │   ├── eval_reliability.py     # reliability signals + gate + fingerprint
│   │   ├── repair_model_c.py       # targeted repair
│   │   ├── gradcam.py              # saliency maps
│   │   ├── external_adni.py        # ADNI external validation
│   │   └── genai/vae.py            # counterfactual stability (VAE)
│   └── notebooks/                # Kaggle notebooks per stage
│
├── track2_brain_tumor/         # Track 2: tumor segmentation (BraTS2020)
│   ├── src/                      # reusable Python modules
│   │   ├── recover_model_a.py      # checkpoint recovery + verification
│   │   ├── eval_model_a_test.py    # held-out test: Dice, IoU, HD95
│   │   ├── train_model_b_unetr.py  # UNETR via MONAI
│   │   ├── modality_removal.py     # FLAIR-removal counterfactual
│   │   └── eval_reliability_seg.py # segmentation reliability
│   └── notebooks/                # Kaggle notebooks per stage
│
├── outputs/                    # logged results, configs, checkpoints (LFS)
├── src/                        # legacy helpers (kept for compatibility)
├── requirements.txt
└── README.md
```

## Verified results

All numbers below come from completed runs. Validation and test results are
labeled. Splits: Track 1 OASIS 36 held-out test scans, 3 seeds; Track 2
BraTS2020 fold0 = 249 train / 63 validation / 56 held-out test cases.

### Track 1 - Alzheimer's classification (test, 36 scans, 3 seeds)

| Model | Accuracy | Balanced acc | Sensitivity | Specificity | F1 | AUC |
|---|---|---|---|---|---|---|
| Model A (ResNet-50) | 0.713 ± 0.013 | 0.706 ± 0.018 | 0.667 ± 0.054 | 0.746 ± 0.022 | 0.658 ± 0.028 | 0.727 ± 0.007 |
| Model B (ViT-B/16) | 0.713 ± 0.013 | - | - | - | - | 0.739 ± 0.012 |

Majority-class baseline accuracy: 0.573.

### Track 1 - Reliability evaluation (slice level, 180 test slices)

| Signal | Model A (ResNet-50) | Model B (ViT-B/16) |
|---|---|---|
| ECE (raw) | 0.0915 | 0.1238 |
| ECE (after temperature scaling) | 0.0829 | 0.0672 |
| Brier score | 0.4358 | 0.4706 |
| Slice accuracy | 0.6111 | 0.5722 |
| Mean OOD score | 8.49e6 (ill-conditioned) | 8252.9 |

- Model disagreement: mismatch fraction 0.394, mean Jensen-Shannon
  divergence 0.0273 (180 slices).
- Reliability gate: degenerate on the fitted validation thresholds (admits
  everything; ungated test error 0.389). Reported as a finding, not hidden.
- Dominant failure fingerprint: overconfident misclassification
  (confidence > 0.8 on errors). Model C repair was not triggered: no
  systematic, repairable subgroup failure beyond overconfidence.
- Model A's raw OOD feature space is ill-conditioned (mean score 8.49e6),
  so that OOD signal is reported as not usable rather than tuned post hoc.

### Track 2 - Brain tumor segmentation (BraTS2020)

Model A (nnU-Net, held-out test, 56 cases):

| Metric | Value |
|---|---|
| Foreground Dice | 0.9005 ± 0.0787 |
| Label 1 (NCR/NET) Dice | 0.6519 ± 0.2970 |
| Label 2 (edema) Dice | 0.7769 ± 0.1696 |
| Label 3 (enhancing tumor) Dice | 0.7804 ± 0.2129 |

Model B (UNETR via MONAI, 96x96x96 patches, 40 epochs, DiceCE loss):

| Metric | Value |
|---|---|
| Best validation Dice | 0.6305 (epoch 40) |
| Validation Dice curve | 0.5765 (ep10) → 0.6069 (ep20) → 0.6159 (ep30) → 0.6305 (ep40) |
| Held-out test (56 cases) | WT Dice 0.7913 / TC Dice 0.6279 / ET Dice 0.6358 (IoU 0.6921 / 0.5181 / 0.5223; HD95 8.86 / 13.32 / 12.73 mm) |
| Validation (63 cases) | WT Dice 0.8188 / TC Dice 0.7104 / ET Dice 0.6803 (HD95 4.95 / 7.01 / 7.02 mm) |
| Reliability vs Model A (56 test cases) | 56/56 cases show disagreement; mean foreground disagreement rate 0.3688; per-class agreement IoU: necrotic 0.3578, edema 0.5938, enhancing 0.5750 |

FLAIR-removal counterfactual (Model A, 56 test cases): mean stability Dice
0.4176; 51 of 56 cases became unstable (whole-tumor Dice ≤ 0.8) when FLAIR
was removed. Model A's predictions depend critically on the FLAIR sequence.

## Transparency: what was run, and what it produced

Every number above comes from a real, logged Kaggle run. The executed
notebooks (with cell outputs, metrics and plots saved) are in
`track1_alzheimers/notebooks/` and `track2_brain_tumor/notebooks/` - they are
the exact committed Kaggle versions that produced these results, downloaded
after the runs completed.

- **Run history, failed/re-run attempts, and every deviation from the original
  plan** (UNETR 40 epochs instead of 200, the lost 12.9-hour training run, the
  257/55 -> 249/63/56 split correction, the degenerate Track 1 gate, the
  ill-conditioned OOD signal, eval v1/v2 failures and fixes): see
  [docs/RUN_HISTORY.md](docs/RUN_HISTORY.md).
- **Raw result data** (per-case records and summary tables used for every
  figure and headline number): [docs/results/](docs/results/).
- **Figures** (vector PDFs generated from that raw data):
  [docs/figures/](docs/figures/).

No result in this repository is hidden or cherry-picked; negative and
degenerate findings are reported as found.

## Data and models

- The training datasets are NOT in this repository. OASIS and BraTS are
  access-controlled; get them from their official sources and accept their
  terms there.
- The trained weights for the demo are published in
  [Releases](https://github.com/pankurajasaivardhan/Neuroshield/releases/tag/demo-v1)
  (weights only, no scan data).
- Data citations: OASIS: Marcus et al., 2007 (doi:10.1162/jocn.2007.19.9.1498).
  BraTS: Menze et al., 2015 (doi:10.1109/TMI.2014.2377694); Bakas et al.,
  2017 (doi:10.1109/TMI.2017.2705472); Bakas et al., 2018
  (arXiv:1811.02629, TCIA doi:10.7937/K9/TCIA.2017.KLXWJJ1Q).

## Known limitations

- External validation on ADNI (Track 1) was not run; it is future work.
- Model A in Track 2 is the friend's 2D nnU-Net checkpoint; a 2D vs 3D
  nnU-Net comparison is an open decision and was not retrained.
- Kaggle free-tier GPU sessions have a wall-clock limit; long trainings are
  capped accordingly (UNETR was trained for 40 epochs, not 60).
- The demo runs on CPU; a full tumor case takes a few minutes.

## Project rules

- Gate thresholds are fit on validation data, never on the test split.
- The Track 2 56-case outer test set stays untouched until final evaluation.
- Always label numbers as validation or test results.
- A failed repair is a valid result; report it.
- Two tracks only: Alzheimer's classification and brain tumor segmentation.

## Run it yourself (demo website)

You do NOT need the training datasets. The trained models and the code are
enough to analyze your own compatible MRI files on your own computer - free,
no accounts, no cloud, CPU only.

What you need:

- A computer with about 4 GB of free memory and 7 GB of free disk
- Python 3.10 or newer. Fully tested end-to-end on Linux (Kaggle, CPU).
  On macOS Apple Silicon the pinned dependency wheels are verified available
  for Python 3.12, but the first full Mac run is still pending - please open
  an issue if something breaks. Windows is not tested - use WSL2.
- Internet for the first run only (downloads ~1.7 GB of models + libraries)

Steps:

1. Get the code: `git clone https://github.com/pankurajasaivardhan/Neuroshield.git`
   or click **Code > Download ZIP** on this page and unzip it.
2. Run the demo:
   - Mac: open the `demo/` folder and double-click **run_mac.command**
     (step-by-step: [demo/SETUP_MAC.md](demo/SETUP_MAC.md))
   - Linux: `bash demo/run_local.sh`
3. The first run downloads the trained models automatically from
   [Releases](https://github.com/pankurajasaivardhan/Neuroshield/releases/tag/demo-v1)
   (3 Alzheimer's checkpoints + 1 tumor model bundle, weights only, no scans).

   **Publication status:** the 3 Alzheimer's checkpoints are live now; the
   tumor model bundle is still being published. Until it lands, the
   Alzheimer's track works fully and the scripts print a specific note
   about the pending tumor file instead of failing.
4. Open http://localhost:8000 in your browser.

Using the website:

- Track 1 (Alzheimer's): upload one structural brain MRI
  (.nii, .nii.gz, or an .hdr/.img pair; OASIS-style T1 scan).
- Track 2 (brain tumor): upload the 4 BraTS sequences of one case
  (FLAIR, T1, T1ce, T2) as .nii or .nii.gz files.
- The site shows the prediction plus reliability signals and can abstain
  when the models disagree or the input looks out of distribution.
  Agreement between models is not proof a prediction is right - treat every
  output as a research demo, never a diagnosis.
