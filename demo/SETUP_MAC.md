# NeuroShield on your Mac - step by step

Runs the demo website on your Mac. No cloud, no accounts, no cost.
People on the same Wi-Fi can open the link it prints; your Mac does the
actual AI inference.

## What you need

- A Mac with Apple Silicon (M1/M2/M3/M4) running macOS 14 or newer
- About 4 GB of free memory while the app runs (the app peaks around 2.4 GB)
- About 7 GB free disk (AI libraries ~1.5 GB + models ~1.7 GB + room to work)
- Internet for the first setup only (downloads ~1.7 GB of models and libraries)

## Step 1 - Install Python 3.12 (once, ~2 min)

1. Go to https://www.python.org/downloads/ and click the big yellow
   "Download Python 3.12.x" button.
2. Open the downloaded .pkg and click Continue / Install through the installer.
3. Open Terminal (Cmd+Space, type "Terminal", Enter) and type:
       python3 --version
   It must say Python 3.12.x. The package is pinned and wheel-checked against
   Python 3.12 on Apple Silicon specifically, so please use 3.12 even if
   another version is already installed.

## Step 2 - Get the code

Either `git clone https://github.com/pankurajasaivardhan/Neuroshield.git`
or on the GitHub page click **Code > Download ZIP** and unzip it.

## Step 3 - Start the demo (~10 min first time, ~10 sec after)

1. Open the `demo/` folder and double-click **run_mac.command**.
   - If macOS says it can't open an unidentified developer: right-click the
     file > Open > Open. (It's a text script - you can read every line.)
2. First run downloads the trained models (~1.7 GB: three Alzheimer's
   checkpoints + the tumor model bundle, from GitHub Releases - weights
   only, no scan data) and the AI libraries. Leave it running; when you see
   "NeuroShield on http://localhost:8000" it's ready.
3. Open http://localhost:8000 in your browser.

## Step 4 - Analyze a scan

- Alzheimer's track: upload one structural brain MRI
  (.nii, .nii.gz, or an .hdr/.img pair; OASIS-style T1 scan).
- Brain tumor track: upload the 4 BraTS sequences of one case
  (FLAIR, T1, T1ce, T2) as .nii or .nii.gz files.
- Results include reliability signals; the site can abstain when a
  prediction is unreliable. Agreement between models is not proof a
  prediction is right - this is a research demo, not a diagnosis.

## Notes

- The full training datasets (OASIS, BraTS) are NOT needed and are not
  downloaded. The tumor bundle is 3 model files (146 MB) extracted from the
  project's private 48.5 GB Kaggle training workspace.
- Stop the app with Ctrl+C in its Terminal window.
- Everything runs on CPU. A full tumor case takes a few minutes.
