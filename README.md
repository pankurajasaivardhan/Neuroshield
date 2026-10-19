# NeuroShield

A two-track medical AI reliability project on brain MRI. NeuroShield does
not only train models; it checks when a prediction can be trusted, records
why a model fails, repairs the specific weakness, and abstains with an
evidence-grounded explanation when a prediction is unreliable.

- Track 1 - Alzheimer's disease classification on OASIS structural MRI
- Track 2 - Brain tumor (glioma) segmentation on BraTS2020

Research prototype. Not a clinically validated diagnostic system.

## Repository structure

```
Neuroshield/
├── shared/                      # Reliability framework used by both tracks
│   ├── reliability/             # uncertainty, calibration, OOD, disagreement,
│   │                            # gate, failure fingerprint, report schema
│   ├── explain/                 # RAG + LLM explanation after abstention
│   └── app/                     # FastAPI backend + Streamlit frontend
│
├── track1_alzheimers/           # Track 1: Alzheimer's classification (OASIS)
│   ├── src/                     # reusable Python modules
│   │   ├── data.py                  # 2.5D OASIS dataset
│   │   ├── train_model_a.py         # ResNet-50 reference baseline
│   │   ├── train_model_b.py         # ViT-B/16 (finalized Model B)
│   │   ├── eval_reliability.py      # reliability signals + gate + fingerprint
│   │   ├── repair_model_c.py        # targeted repair
│   │   ├── gradcam.py               # saliency maps
│   │   ├── external_adni.py         # ADNI external validation
│   │   └── genai/vae.py             # counterfactual stability (VAE)
│   └── notebooks/               # run these on Kaggle, in number order
│       ├── 01_explore_oasis.ipynb
│       ├── 02_prepare_dataset.ipynb
│       ├── 03_train_model_b_final.ipynb
│       ├── 04_train_model_a.ipynb
│       ├── 05_reliability_evaluation.ipynb
│       ├── 06_repair_model_c.ipynb
│       ├── 07_vae_counterfactual.ipynb
│       └── 08_external_adni.ipynb
│
├── track2_brain_tumor/          # Track 2: tumor segmentation (BraTS2020)
│   ├── src/                     # reusable Python modules
│   │   ├── recover_model_a.py       # checkpoint recovery + verification
│   │   ├── eval_model_a_test.py     # held-out test: Dice, IoU, HD95
│   │   ├── train_model_b_unetr.py   # UNETR via MONAI
│   │   ├── modality_removal.py      # FLAIR-removal counterfactual
│   │   └── eval_reliability_seg.py  # segmentation reliability
│   └── notebooks/               # run these on Kaggle, in number order
│       ├── braintumor-01-final.ipynb
│       ├── braintumor-02-recover-model-a.ipynb
│       ├── braintumor-03-test-evaluation.ipynb
│       ├── braintumor-04-train-unetr.ipynb
│       ├── braintumor-05-modality-removal.ipynb
│       └── braintumor-06-reliability-evaluation.ipynb
│
├── outputs/                     # logged results, configs, checkpoints (LFS)
├── src/                         # legacy helpers (kept for compatibility)
├── requirements.txt
└── README.md
```

## Current status

| Stage | Track 1 (Alzheimer's) | Track 2 (Brain tumor) |
|---|---|---|
| Dataset preparation | Done | Done |
| Model A | Pending | Training done, recovery + test eval pending |
| Model B | Done (test accuracy 0.713 +/- 0.013, AUC 0.739 +/- 0.012, 3 seeds) | Pending (UNETR) |
| Reliability evaluation | Code ready, run pending | Code ready, run pending |
| Failure analysis / Model C | Code ready, run pending | Code ready, run pending |
| External / domain shift | ADNI pending | Pending |
| Reliability Gate | Code ready, run pending | Code ready, run pending |

Only the numbers above are verified results. Everything else reports as
pending until its notebook has actually been run.

## How to run

Notebooks run on Kaggle (free GPU). Each new notebook clones this repo in
its first cell, installs `requirements.txt`, and calls the modules under
`shared/` and the track's `src/`.

Track 2 first (secure the trained Model A before any new training):
`braintumor-02` recovers the artifact, `braintumor-03` computes the real
held-out test metrics, `braintumor-04` trains UNETR, `05` and `06` run the
counterfactual and reliability studies.

Track 1: `04` trains the reference baseline, `05` runs the reliability
evaluation and gate, `06` repairs only if a systematic failure was found,
`07` and `08` are the generative and external-validation stages.

Local app (after a track has a checkpoint and a fitted gate):

```
pip install -r requirements.txt
uvicorn shared.app.backend:app
streamlit run shared/app/frontend.py
```

## Project rules

- Gate thresholds are fit on validation data, never on the test split.
- The Track 2 56-case outer test set stays untouched until final evaluation.
- Always label numbers as validation or test results.
- A failed repair is a valid result; report it.
- Two tracks only: Alzheimer's classification and brain tumor segmentation.
