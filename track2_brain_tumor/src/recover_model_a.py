# Track 2 Model A artifact recovery + verification (never retrains).
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
MIN_CKPT_BYTES = 100 * 1024 * 1024

def trainer_dir(root: str, dataset: str, trainer: str, config: str) -> str:
    return os.path.join(root, dataset, f'{trainer}__nnUNetPlans__{config}')

def verify_structure(tdir: str, fold: int) -> dict:
    fold_dir = os.path.join(tdir, f'fold_{fold}')
    checks = {}
    for (name, path) in [('checkpoint_best.pth', os.path.join(fold_dir, 'checkpoint_best.pth')), ('checkpoint_final.pth', os.path.join(fold_dir, 'checkpoint_final.pth')), ('plans.json', os.path.join(tdir, 'plans.json')), ('dataset.json', os.path.join(tdir, 'dataset.json'))]:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        ok = exists and (size > MIN_CKPT_BYTES if name.endswith('.pth') else size > 100)
        checks[name] = {'path': path, 'exists': exists, 'bytes': size, 'ok': ok}
    return checks

def repair_structure(tdir: str, dataset: str, root: str) -> list[str]:
    actions = []
    ds_json_target = os.path.join(tdir, 'dataset.json')
    if not os.path.exists(ds_json_target):
        src = os.path.join(os.path.dirname(os.path.dirname(root)), 'nnUNet_raw', dataset, 'dataset.json')
        if os.path.exists(src):
            shutil.copy2(src, ds_json_target)
            actions.append(f'copied dataset.json from {src}')
        else:
            actions.append('dataset.json NOT FOUND in nnUNet_raw; regenerate from the manifest before inference')
    return actions

def fresh_load_test(ckpt: str) -> bool:
    code = f"import torch, sys; sd = torch.load({ckpt!r}, map_location='cpu', weights_only=False); print(sorted(sd.keys())[:5] if isinstance(sd, dict) else type(sd)); sys.exit(0)"
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
    return r.returncode == 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nnunet-results', required=True)
    ap.add_argument('--dataset-name', default='Dataset001_BraTS2020')
    ap.add_argument('--trainer', default='nnUNetTrainer_100epochs')
    ap.add_argument('--config', default='2d')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    tdir = trainer_dir(args.nnunet_results, args.dataset_name, args.trainer, args.config)
    if not os.path.isdir(tdir):
        raise SystemExit(f'Trainer directory not found: {tdir}. Check --nnunet-results / names against the Kaggle session.')
    checks = verify_structure(tdir, args.fold)
    print(json.dumps(checks, indent=2))
    actions = repair_structure(tdir, args.dataset_name, args.nnunet_results)
    for a in actions:
        print('[repair]', a)
    checks = verify_structure(tdir, args.fold)
    missing = [k for (k, v) in checks.items() if not v['ok']]
    if missing:
        raise SystemExit(f'Artifact incomplete, missing/invalid: {missing}. Do NOT retrain automatically; escalate to the project owner.')
    dest = os.path.join(args.out, 'model', args.dataset_name, f'{args.trainer}__nnUNetPlans__{args.config}')
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(tdir, dest)
    print(f'[persist] copied to {dest}')
    best = os.path.join(dest, f'fold_{args.fold}', 'checkpoint_best.pth')
    ok = fresh_load_test(best)
    meta = {'trainer': args.trainer, 'config': args.config, 'fold': args.fold, 'source': tdir, 'artifact': dest, 'checks': checks, 'repair_actions': actions, 'fresh_load_ok': ok}
    with open(os.path.join(args.out, 'model_a_artifact_verification.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('fresh-session load:', 'OK' if ok else 'FAILED')
    if not ok:
        raise SystemExit('Checkpoint did not load in a fresh process. Do not mark Model A complete.')
if __name__ == '__main__':
    main()
