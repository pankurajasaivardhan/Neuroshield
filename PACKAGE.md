# NeuroShield

Reliability layer for medical AI on brain MRI. Two tracks: Alzheimer's
classification (OASIS) and brain tumor segmentation (BraTS2020). The system
checks when a prediction can be trusted, records why models fail, repairs
the weakness, and abstains with an evidence-grounded explanation when a
prediction is unreliable. Research prototype, not a clinical device.

## Structure

```
neuroshield/
  shared/
    reliability/     uncertainty, calibration, OOD, disagreement, gate,
                     fingerprint, report
    explain/rag.py   literature retrieval + LLM explanation after abstention
    app/             FastAPI backend + Streamlit frontend
  track1/            Alzheimer's classification (OASIS)
    data.py                2.5D dataset
    train_model_a.py       ResNet-50 reference baseline
    train_model_b.py       ViT-B/16 (matches the finalized notebook)
    eval_reliability.py    reliability signals + gate + fingerprint
    repair_model_c.py      targeted repair
    gradcam.py             saliency maps
    external_adni.py       ADNI external validation
    genai/vae.py           counterfactual stability via latent perturbation
  track2/            Brain tumor segmentation (BraTS2020)
    recover_model_a.py     checkpoint recovery + verification
    eval_model_a_test.py   held-out test metrics (Dice, IoU, HD95)
    train_model_b_unetr.py UNETR via MONAI
    modality_removal.py    modality-removal counterfactual
    eval_reliability_seg.py segmentation reliability
```

## Status

Done and verified: Track 1 Model B (ViT-B/16) test results, accuracy
0.713 +/- 0.013 and ROC AUC 0.739 +/- 0.012 over 3 seeds on 36 held-out
scans. Track 2 Model A (2D nnU-Net) training with internal validation Dice
0.7651. The shared reliability core is unit-tested on synthetic data.

Everything else is pending real runs: Track 1 Model A, Track 2 test metrics
and UNETR, reliability measurements, Model C repair, gate evaluation, the
app, and external validation. GPU training and Kaggle steps run on
Kaggle/Colab; each script says so in its first line.

## Run order

Track 2 first (secure Model A before anything expensive):

1. track2/recover_model_a.py on the Kaggle session with the training output
2. Fix the metrics env: fresh session, pip install -U scipy numpy, verify
   "from scipy.ndimage import distance_transform_edt" imports
3. track2/eval_model_a_test.py for the real test metrics
4. Owner decision on 2D vs 3d_fullres before any retrain
5. track2/train_model_b_unetr.py, then modality_removal.py, then
   eval_reliability_seg.py

Track 1:

1. track1/train_model_a.py (3 seeds)
2. track1/eval_reliability.py (needs Model A and B checkpoints)
3. track1/repair_model_c.py only if a fingerprint shows systematic failure
4. track1/genai/vae.py, then track1/external_adni.py after ADNI access

App (after one track has a checkpoint and a fitted gate):

1. uvicorn neuroshield.shared.app.backend:app
2. Register track runners (see register_track in backend.py)
3. streamlit run neuroshield/shared/app/frontend.py

## Rules

- Thresholds are fit on validation data, never on the test split.
- The Track 2 56-case outer test set stays untouched until final evaluation.
- Always label numbers as validation or test.
- A failed repair is a valid result; report it.
- No clinical claims.

## Setup

pip install -r requirements.txt
