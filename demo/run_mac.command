#!/usr/bin/env bash
# NeuroShield - run the demo website on this Mac (free, no cloud account).
# Double-click this file. Serves on http://localhost:8000 (and your Wi-Fi IP,
# so people on the same network can use it). CPU only; needs ~4 GB free RAM.
set -euo pipefail
cd "$(dirname "$0")"

# Python 3.10+ required (Python 3.12 from python.org recommended)
if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
  echo "Python 3.10 or newer not found."
  echo "Install Python 3.12 from https://www.python.org/downloads/ then double-click this file again."
  read -r -p "Press return to close."
  exit 1
fi

ARTS="${NEUROSHIELD_ARTIFACTS:-$PWD/artifacts}"
mkdir -p "$ARTS/track1" "$ARTS/track2/model"
REL="https://github.com/pankurajasaivardhan/Neuroshield/releases/download/demo-v1"

# --- Track 1 checkpoints (3 x 343 MB, GitHub Release) ---
for seed in 42 123 2024; do
  f="$ARTS/track1/model_b_vit_seed${seed}_best.pt"
  [ -s "$f" ] || { echo "downloading Alzheimer's model seed $seed (343 MB)..."; curl -fL --retry 3 -o "$f" "$REL/model_b_vit_seed${seed}_best.pt"; }
done

# --- Track 2 nnU-Net model (model-only bundle, 146 MB, GitHub Release) ---
# NOTE: the full Kaggle dataset is 48.5 GB (contains raw training data) and is
# never downloaded. Only these 3 model files are needed.
M="$ARTS/track2/model/Dataset001_BraTS2020/nnUNetTrainer_100epochs__nnUNetPlans__2d"
if [ ! -f "$M/fold_0/checkpoint_final.pth" ]; then
  z=$(mktemp -d)/neuroshield-track2-weights.zip
  echo "downloading tumor model bundle (146 MB)..."
  if curl -fL --retry 3 -o "$z" "$REL/neuroshield-track2-weights.zip"; then
    # Safety: the bundle must be small and contain ONLY model files - never
    # accept the full 48.5 GB dataset or anything with raw scans.
    sz=$(stat -f%z "$z" 2>/dev/null || echo 0)
    if [ "$sz" -gt 2000000000 ]; then
      echo "ERROR: downloaded bundle is over 2 GB - refusing to use it."; exit 1
    fi
    if unzip -l "$z" | grep -qiE '\.nii|\.nii\.gz|imagesTr|imagesTs|nnUNet_raw'; then
      echo "ERROR: bundle contains raw scan data - refusing to use it."; exit 1
    fi
    unzip -l "$z" | grep -qi 'checkpoint_final.pth' || { echo "ERROR: bundle has no checkpoint_final.pth."; exit 1; }
    tmp=$(mktemp -d)
    (cd "$tmp" && unzip -q "$z")
    mkdir -p "$M/fold_0"
    find "$tmp" -name 'plans.json' -exec cp {} "$M/" \;
    find "$tmp" -name 'dataset.json' -exec cp {} "$M/" \;
    find "$tmp" -iname 'checkpoint_final.pth' -exec cp {} "$M/fold_0/" \;
    rm -rf "$tmp"
    [ -f "$M/fold_0/checkpoint_final.pth" ] || { echo "ERROR: unpack failed"; exit 1; }
  else
    echo "NOTE: could not download the tumor model bundle"
    echo "  (neuroshield-track2-weights.zip from release demo-v1). Check your"
    echo "  internet connection and try again. The Alzheimer's track works"
    echo "  without it - double-click this file again later to add tumor analysis."
  fi
fi

# --- deps + server ---
python3 -m pip install -q -r deploy/local/requirements-mac.txt
export NEUROSHIELD_ARTIFACTS="$ARTS"
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "unavailable")
echo ""
echo "NeuroShield on http://localhost:${PORT:-8000}  (same-Wi-Fi: http://${LAN_IP}:${PORT:-8000})"
echo "Keep this window open while demoing. Ctrl+C to stop."
exec python3 -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --app-dir server
