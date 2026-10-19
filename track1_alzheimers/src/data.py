# Track 1 (Alzheimer's, OASIS): 2.5D dataset, 5 sagittal slices per scan.
from __future__ import annotations
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SLICE_OFFSETS = (-30, -15, 0, 15, 30)

def expand_scans_to_slices(df: pd.DataFrame, offsets=SLICE_OFFSETS) -> pd.DataFrame:
    rows = []
    for (_, r) in df.iterrows():
        for off in offsets:
            rows.append({**r.to_dict(), 'slice_offset': off})
    return pd.DataFrame(rows)

def normalize_mri_slice(stack: np.ndarray) -> np.ndarray:
    nonzero = stack[stack > 0]
    if len(nonzero) == 0:
        return stack
    (p1, p99) = np.percentile(nonzero, [1, 99])
    stack = np.clip(stack, p1, p99)
    stack = (stack - p1) / (p99 - p1 + 1e-08)
    return np.clip(stack, 0, 1).astype(np.float32)

class OASISDataset(Dataset):

    def __init__(self, csv_path: str, expand: bool=True):
        df = pd.read_csv(csv_path)
        self.df = expand_scans_to_slices(df) if expand else df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = nib.load(row['scan_path'])
        volume = img.get_fdata().squeeze()
        center = volume.shape[0] // 2
        off = int(row.get('slice_offset', 0))
        slice_2d = volume[center + off, :, :]
        slice_2d = normalize_mri_slice(slice_2d.astype(np.float32))
        t = torch.from_numpy(slice_2d)[None, None]
        t = torch.nn.functional.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)[0, 0]
        t = (t - torch.tensor(IMAGENET_MEAN[0])) / torch.tensor(IMAGENET_STD[0])
        img3 = t.repeat(3, 1, 1)
        return (img3, torch.tensor(int(row['label']), dtype=torch.long), str(row['ID']))

def scan_level_predictions(model, dataset: OASISDataset, device: str='cpu'):
    model.eval()
    out: dict[str, dict] = {}
    with torch.no_grad():
        for i in range(len(dataset)):
            (img, label, sid) = dataset[i]
            logits = model(img.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            e = out.setdefault(sid, {'probs': [], 'label': int(label)})
            e['probs'].append(probs)
    return {sid: {'probs': np.mean(v['probs'], axis=0), 'label': v['label']} for (sid, v) in out.items()}
