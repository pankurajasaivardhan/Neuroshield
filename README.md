# NeuroShield

A two-track medical AI reliability project on brain MRI. NeuroShield does
not only train models; it checks when a prediction can be trusted, records
why a model fails, repairs the specific weakness, and abstains with an
evidence-grounded explanation when a prediction is unreliable.

- Track 1 - Alzheimer's disease classification on OASIS structural MRI
- Track 2 - Brain tumor (glioma) segmentation on BraTS2020

Research prototype. Not a clinically validated diagnostic system. Not for
medical diagnosis.

Everything you need is in this one file: setup and run, what the models do,
verified results, run history, limitations and citations.

**Contents:** [Run it yourself](#run-it-yourself-demo-website) -
[Step-by-step on macOS](#step-by-step-on-macos-apple-silicon) -
[Using the website](#using-the-website) -
[Repository structure](#repository-structure) -
[Verified results](#verified-results) -
[Run history and transparency](#run-history-and-transparency) -
[Data and models](#data-and-models) -
[Known limitations](#known-limitations) -
[Project rules](#project-rules)

## Run it yourself (demo website)

You do NOT need the training datasets. The trained models and the code are
enough to analyze your own compatible MRI files on your own computer - free,
no accounts, no cloud, CPU only.

What you need:

- A computer with about 4 GB of free memory and 7 GB of free disk
- Python 3.10 or newer (macOS Apple Silicon: Python 3.12 exactly - see below).
  Fully tested end-to-end on Linux (Kaggle, CPU). On macOS Apple Silicon the
  pinned dependency wheels are verified available for Python 3.12, but the
  first full Mac run is still pending - please open an issue if something
  breaks. Windows is not tested - use WSL2.
- Internet for the first run only (downloads ~1.7 GB of models + libraries)

Steps:

1. Get the code: `git clone https://github.com/pankurajasaivardhan/Neuroshield.git`
   or click **Code > Download ZIP** on this page and unzip it.
2. Run the demo:
   - Mac: open the `demo/` folder and double-click **run_mac.command**
     (full step-by-step below)
   - Linux: `bash demo/run_local.sh`
3. The first run downloads the trained models automatically from
   [Releases](https://github.com/pankurajasaivardhan/Neuroshield/releases/tag/demo-v1)
   (3 Alzheimer's checkpoints + 1 tumor model bundle, weights only, no scans).
4. Open http://localhost:8000 in your browser. Stop the app with Ctrl+C in
   its terminal window.

## Step-by-step on macOS (Apple Silicon)

No cloud, no accounts, no cost. People on the same Wi-Fi can open the link
the app prints; your Mac does the actual AI inference.

You need: a Mac with Apple Silicon (M1/M2/M3/M4) on macOS 14 or newer,
~4 GB free memory while the app runs (it peaks around 2.4 GB), ~7 GB free
disk, and internet for the first setup only.

1. Install Python 3.12 (once, ~2 min):
   - Go to https://www.python.org/downloads/ and click the big yellow
     "Download Python 3.12.x" button.
   - Open the downloaded .pkg and click Continue / Install through the
     installer.
   - Open Terminal (Cmd+Space, type "Terminal", Enter) and type
     `python3 --version`. It must say Python 3.12.x. The package is pinned
     and wheel-checked against Python 3.12 on Apple Silicon specifically,
     so use 3.12 even if another version is already installed.
2. Get the code (as above) and open the `demo/` folder.
3. Double-click **run_mac.command** (~10 min the first time, ~10 sec after):
   - If macOS says it can't open an unidentified developer: right-click the
     file > Open > Open. (It's a text script - you can read every line.)
   - The first run downloads the trained models (~1.7 GB, from GitHub
     Releases - weights only, no scan data) and the AI libraries. Leave it
     running; when you see "NeuroShield on http://localhost:8000" it's ready.
4. Open http://localhost:8000 in your browser.

## Using the website

- Track 1 (Alzheimer's): upload one structural brain MRI
  (.nii, .nii.gz, or an .hdr/.img pair; OASIS-style T1 scan).
- Track 2 (brain tumor): upload the 4 BraTS sequences of one case
  (FLAIR, T1, T1ce, T2) as .nii or .nii.gz files.
- The site shows the prediction plus reliability signals and can abstain
  when the models disagree or the input looks out of distribution.
  Agreement between models is not proof a prediction is right - treat every
  output as a research demo, never a diagnosis.
- One interface, two separate models. Everything runs on CPU; a full tumor
  case takes a few minutes.

## Repository structure

```
Neuroshield/
├── demo/                       # Self-contained demo website (run this)
│   ├── server/                   # FastAPI backend + model runners + frontend
│   ├── deploy/local/             # pinned CPU requirements (Linux + macOS)
│   ├── run_mac.command           # double-click start for macOS
│   └── run_local.sh              # start script for Linux
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
│   └── notebooks/                # Kaggle notebooks per stage (executed)
│
├── track2_brain_tumor/         # Track 2: tumor segmentation (BraTS2020)
│   ├── src/                      # reusable Python modules
│   │   ├── recover_model_a.py      # checkpoint recovery + verification
│   │   ├── eval_model_a_test.py    # held-out test: Dice, IoU, HD95
│   │   ├── train_model_b_unetr.py  # UNETR via MONAI
│   │   ├── modality_removal.py     # FLAIR-removal counterfactual
│   │   └── eval_reliability_seg.py # segmentation reliability
│   └── notebooks/                # Kaggle notebooks per stage (executed)
│
├── docs/results/               # raw per-run data behind every number
├── docs/figures/               # vector PDF figures generated from that data
├── outputs/                    # logged results, configs, checkpoints (LFS)
├── requirements.txt
└── README.md                   # this file - the only guide you need
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

## Run history and transparency

Every number above comes from a real, logged Kaggle run. The executed
notebooks (with cell outputs, metrics and plots saved) are in
`track1_alzheimers/notebooks/` and `track2_brain_tumor/notebooks/` - they are
the exact committed Kaggle versions that produced these results, downloaded
after the runs completed. Nothing is cherry-picked; negative and degenerate
results are reported as found.

### What was actually run (final state, 2026-10-21)

All runs were executed on Kaggle under two accounts (Track 1: `mohanroyal273`,
Track 2: `rajasaivardhan`).

Track 1 - Alzheimer's classification (OASIS-1):

| Run | Notebook version | Result |
|---|---|---|
| Model A training (ResNet-50, lr=1e-5, 3 seeds) | `track1-model-a-resnet50-baseline` v4 | scan-level acc 0.7130 ± 0.0131, balanced 0.7063 ± 0.0184, sens 0.6667 ± 0.0544, spec 0.7460 ± 0.0224, F1 0.6583 ± 0.0279, AUC 0.7270 ± 0.0069 (36 test scans) |
| Model B training (ViT-B/16, 3 seeds) | `03-train-model-b-final` v1 (executed 2026-09-24) | scan acc 0.7130 ± 0.0131, AUC 0.7386 ± 0.0120; majority baseline 0.5732 |
| Reliability evaluation | `notebooke0cf27f81b` v3 | slice-level, 180 test slices - see the reliability table above |

Track 1 reliability extras: disagreement mismatch fraction 0.3944, mean JS
divergence 0.0273 (180 slices). Gate: DEGENERATE - validation-fitted
thresholds admit every input (cutoff 1.00), so accepted error equals ungated
test error 0.3889. The threshold-swept risk-coverage curve is NOT flat
(accepted-subset error ~0.18-0.20 at 15-35% coverage), i.e. the reliability
score's ranking carries signal that the fitted thresholds cannot exploit at
this cohort size. Failure fingerprint `fp-25b871fd`: overconfident
misclassification (errors at confidence > 0.8).

Track 2 - Brain tumor segmentation (BraTS 2020). Split (fold0,
authoritative): 249 train / 63 val / 56 test, from `splits_final.json` under
`nnUNet_preprocessed`.

| Run | Notebook version | Result |
|---|---|---|
| Model A training (2D nnU-Net) | `braintumor-01-final` (latest committed) | see test metrics above |
| Model A held-out test | `track2-model-a-test-metrics` v5 | 56 cases: foreground Dice 0.9005 ± 0.0787; per-label Dice l1 0.6519 ± 0.2970, l2 0.7769 ± 0.1696, l3 0.7804 ± 0.2129 |
| Model B training (UNETR, MONAI, 96³ patches, DiceCE, 40 epochs) | `track2-model-b-unetr` v5 | best val mean Dice 0.6305 (epoch 40); curve 0.5765 / 0.6069 / 0.6159 / 0.6305 at epochs 10/20/30/40 |
| FLAIR-removal counterfactual (Model A) | `track2-flair-removal` v6 | 56/56 cases: mean stability Dice 0.4176; 51/56 unstable (WT Dice ≤ 0.8) |
| Model B eval + reliability | `notebookc85717a0d1` v3 | UNETR test (n=56): WT Dice 0.7913, TC 0.6279, ET 0.6358 (HD95 8.86 / 13.32 / 12.73 mm); val (n=63): WT 0.8188, TC 0.7104, ET 0.6803. A-vs-B agreement (test): mean foreground disagreement 0.3688; per-class IoU necrotic 0.3578 / edema 0.5938 / enhancing 0.5750 |

### Deviations from the original plan (honest notes)

1. **UNETR trained for 40 epochs, not 200.** The training script's original
   default was 200 epochs. The 40-epoch run was kept as the final Model B
   (validation Dice was still improving slowly at epoch 40: 0.6159 → 0.6305).
   Repo defaults were updated to match what was actually run.
2. **A 12.9-hour UNETR training run was lost.** The likely cause is Kaggle's
   ~12-hour GPU session wall (inferred, not confirmed; the session died and
   its outputs were not persisted). The final v5 run (9h48m) completed within
   the window.
3. **Split string correction 257/55 → 249/63/56.** Older notebook markdown
   (including the executed `braintumor-01-final` STAGE-3 heading) still says
   257/55/56. That string is stale: the split actually used by every reported
   result is fold0 = 249/63/56 from `splits_final.json`. The executed
   notebooks are kept as they really ran; the correction is documented here
   rather than edited into history.
4. **Track 2 reliability ran on the test split, not validation.** Model A
   validation predictions were never persisted (the `neuroshield-track2-model-a`
   dataset contains only `test_predictions/`). The segmentation reliability
   script fits no thresholds, so running it on the 56-case test set introduces
   no supervision leak. Eval attempts v1 (failed fast on the missing validation
   predictions) and v2 (failed on a CUDA-tensor `.numpy()` staging bug)
   preceded the successful v3; both failures were diagnosed and fixed in the
   open.
5. **Track 1 gate degenerates (negative result).** Validation-fitted accept
   thresholds admit all 180 test slices. Reported as found; no gate is claimed.
6. **Track 1 Model A OOD signal is ill-conditioned.** Mahalanobis OOD scores
   (~8.5e6) reflect a degenerate feature covariance on this cohort, not a
   tuned detector. Reported as an honest negative measurement.
7. **Model C repair was not triggered in either track.** Track 1's audit
   found overconfident misclassification without a repairable subgroup;
   retraining to chase calibration on 36 test scans would fit noise. Recorded
   as a valid negative repair outcome per the project rules.
8. **ADNI external evaluation was not run** (GPU quota). Documented as a
   limitation, not a result.
9. **2D vs 3D nnU-Net for Model A** remains an open decision; no retrain was
   done. Voxel-level calibration for segmentation was not possible because
   probability maps were not saved during prediction (`--save-probabilities`).
10. **Track 1 reliability metrics are slice-level** (180 test slices), while
    the headline classification metrics are scan-level (36 scans, 3 seeds).
    They answer different questions and are labeled as such wherever reported.

### Raw data and figures

- **Raw result data** (per-case records and summary tables used for every
  figure and headline number): [docs/results/](docs/results/).
- **Figures** (vector PDFs generated from that raw data):
  [docs/figures/](docs/figures/) - `fig_unetr_val_curve.pdf` (UNETR
  validation Dice by epoch), `fig_flair_stability.pdf` (FLAIR-removal
  stability distribution), `fig_track1_risk_coverage.pdf` (Track 1
  risk-coverage sweep), `fig_track2_agreement_iou.pdf` (Track 2 per-case
  agreement IoU by class), `fig_track1_js_divergence.pdf` (Track 1
  JS-divergence distribution).

## Data and models

- The training datasets are NOT in this repository, and no MRI scans are
  bundled anywhere in it. OASIS and BraTS are access-controlled; get them
  from their official sources and accept their terms there. The OASIS-1 Data
  Use Terms do not allow redistributing the data to third parties without
  written permission, and the BraTS data usage terms do not grant
  redistribution rights. Upload your own compatible scans in the demo.
- The trained weights for the demo are published in
  [Releases](https://github.com/pankurajasaivardhan/Neuroshield/releases/tag/demo-v1)
  (weights only, no scan data). The tumor bundle is 3 model files (146 MB)
  extracted from the project's private 48.5 GB Kaggle training workspace.
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
