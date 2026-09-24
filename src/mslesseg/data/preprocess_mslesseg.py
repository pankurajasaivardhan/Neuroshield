from __future__ import annotations
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd
MODALITIES = ["T1", "T2", "FLAIR"]
def load_nifti(path: Path):
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    return image, data
def zscore_nonzero(data: np.ndarray) -> np.ndarray:
    """
    Normalize MRI intensities using only non-zero voxels.
    Background remains zero.
    """
    mask = data != 0
    if not np.any(mask):
        return data.astype(np.float32)
    values = data[mask]
    mean = values.mean()
    std = values.std()
    if std < 1e-8:
        return np.zeros_like(data, dtype=np.float32)
    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[mask] = (values - mean) / std
    return normalized
def preprocess_scan(row: pd.Series, output_dir: Path):
    """
    Preprocess one MSLesSeg scan.
    The original NIfTI files are never modified.
    Output:
        T1.npy
        T2.npy
        FLAIR.npy
        MASK.npy
    """
    scan_dir = Path(row["dir"])
    patient = row["patient"]
    timepoint = row["timepoint"]
    output_case = output_dir / patient / str(timepoint)
    output_case.mkdir(parents=True, exist_ok=True)
    # Locate files
    if timepoint == "test":
        filenames = {
            "T1": f"{patient}_T1.nii.gz",
            "T2": f"{patient}_T2.nii.gz",
            "FLAIR": f"{patient}_FLAIR.nii.gz",
            "MASK": f"{patient}_MASK.nii.gz",
        }
    else:
        filenames = {
            "T1": f"{patient}_{timepoint}_T1.nii.gz",
            "T2": f"{patient}_{timepoint}_T2.nii.gz",
            "FLAIR": f"{patient}_{timepoint}_FLAIR.nii.gz",
            "MASK": f"{patient}_{timepoint}_MASK.nii.gz",
        }
    # Load and normalize MRI modalities
    for modality in MODALITIES:
        path = scan_dir / filenames[modality]
        _, data = load_nifti(path)
        data = zscore_nonzero(data)
        np.save(
            output_case / f"{modality}.npy",
            data.astype(np.float32),
        )
    # Load mask
    mask_path = scan_dir / filenames["MASK"]
    _, mask = load_nifti(mask_path)
    # Convert mask to binary
    mask = (mask > 0).astype(np.uint8)
    np.save(
        output_case / "MASK.npy",
        mask,
    )
def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MSLesSeg using the verified manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/mslesseg_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/processed/mslesseg"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N scans for testing.",
    )
    args = parser.parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {args.manifest}"
        )
    df = pd.read_csv(args.manifest)
    # Only verified scans
    df = df[df["ok"] == True].copy()
    if df.empty:
        raise RuntimeError(
            "No verified scans found in manifest."
        )
    if args.limit is not None:
        df = df.head(args.limit)
    print("MSLesSeg preprocessing")
    print(f"Scans to process: {len(df)}")
    print(f"Output directory: {args.output_dir}")
    print()
    for index, row in df.iterrows():
        patient = row["patient"]
        timepoint = row["timepoint"]
        print(
            f"[{index + 1}/{len(df)}] "
            f"{patient} / {timepoint}"
        )
        preprocess_scan(
            row,
            args.output_dir,
        )
    print()
    print("PREPROCESSING COMPLETE")
if __name__ == "__main__":
    main()