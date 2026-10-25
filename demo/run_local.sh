#!/usr/bin/env bash
# NeuroShield - run the demo website locally on Linux (free, no cloud account).
# Serves on http://localhost:8000 (and your LAN IP). CPU only, ~4 GB free RAM.
set -euo pipefail
cd "$(dirname "$0")"
ARTS="${NEUROSHIELD_ARTIFACTS:-$PWD/artifacts}"
mkdir -p "$ARTS/track1" "$ARTS/track2/model"
REL="https://github.com/pankurajasaivardhan/Neuroshield/releases/download/demo-v1"

# --- Track 1 checkpoints (3 x 343 MB, GitHub Release) ---
for seed in 42 123 2024; do
  f="$ARTS/track1/model_b_vit_seed${seed}_best.pt"
  [ -s "$f" ] || { echo "downloading Alzheimer's model seed $seed (343 MB)..."; curl -fL --retry 3 -o "$f" "$REL/model_b_vit_seed${seed}_best.pt"; }
done

# --- Track 2 nnU-Net model (model-only bundle, 146 MB, GitHub Release) ---
M="$ARTS/track2/model/Dataset001_BraTS2020/nnUNetTrainer_100epochs__nnUNetPlans__2d"
if [ ! -f "$M/fold_0/checkpoint_final.pth" ]; then
  z=$(mktemp -d)/neuroshield-track2-weights.zip
  echo "downloading tumor model bundle (146 MB)..."
  if curl -fL --retry 3 -o "$z" "$REL/neuroshield-track2-weights.zip"; then
    sz=$(stat -c%s "$z" 2>/dev/null || echo 0)
    if [ "$sz" -gt 2000000000 ]; then echo "ERROR: bundle over 2 GB - refusing."; exit 1; fi
    if unzip -l "$z" | grep -qiE '\.nii|\.nii\.gz|imagesTr|imagesTs|nnUNet_raw'; then
      echo "ERROR: bundle contains raw scan data - refusing."; exit 1
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
    echo "NOTE: the tumor model bundle is still being published to the release"
    echo "  (file neuroshield-track2-weights.zip under demo-v1). The Alzheimer's"
    echo "  track works without it - re-run this script later to add tumor analysis."
  fi
fi

# --- deps + server ---
python3 -m pip install -q "torch==2.14.1+cpu" "torchvision==0.29.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -q -r deploy/local/requirements.txt
export NEUROSHIELD_ARTIFACTS="$ARTS"
echo "NeuroShield on http://localhost:${PORT:-8000} (LAN: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${PORT:-8000})"
exec python3 -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --app-dir server
