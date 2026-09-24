from __future__ import annotations
import argparse
from pathlib import Path
import nibabel as nib
import pandas as pd
MODALITIES = ["T1", "T2", "FLAIR"]
EXPECTED_SHAPE = (182, 218, 182)
EXPECTED_PATIENTS = 75
EXPECTED_SCANS = 115
def load_nifti(path: Path):
    return nib.load(str(path))
def check_scan(
    scan_dir: Path,
    patient: str,
    timepoint: str,
    file_map: dict[str, Path],
):
    row = {
        "dir": str(scan_dir),
        "patient": patient,
        "timepoint": timepoint,
        "t1_shape": None,
        "t2_shape": None,
        "flair_shape": None,
        "mask_shape": None,
        "issues": "",
        "ok": False,
    }
    issues = []
    # Check all required files
    for modality in MODALITIES + ["MASK"]:
        path = file_map.get(modality)
        if path is None or not path.exists():
            issues.append(f"missing {modality}")
    if issues:
        row["issues"] = "; ".join(issues)
        return row
    # Load and check shapes
    loaded = {}
    for modality, path in file_map.items():
        try:
            loaded[modality] = load_nifti(path)
        except Exception as exc:
            issues.append(f"{modality} failed to load: {exc}")
    if issues:
        row["issues"] = "; ".join(issues)
        return row
    for modality, image in loaded.items():
        shape = tuple(image.shape)
        row[f"{modality.lower()}_shape"] = shape
        if shape != EXPECTED_SHAPE:
            issues.append(
                f"{modality} shape {shape} != expected {EXPECTED_SHAPE}"
            )
    # Check that all modalities have identical shapes
    image_shapes = [
        tuple(loaded[m].shape)
        for m in MODALITIES + ["MASK"]
    ]
    if len(set(image_shapes)) != 1:
        issues.append("T1/T2/FLAIR/MASK shapes do not match")
    row["issues"] = "; ".join(issues)
    row["ok"] = len(issues) == 0
    return row
def find_training_scans(data_dir: Path):
    """
    Training layout:
    train/
      P01/
        T1/
        T2/
        T3/
      P02/
        T1/
        ...
    Each timepoint folder is one scan.
    """
    scans = []
    train_dir = data_dir / "train"
    if not train_dir.exists():
        return scans
    for patient_dir in sorted(train_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient = patient_dir.name
        for timepoint_dir in sorted(patient_dir.iterdir()):
            if not timepoint_dir.is_dir():
                continue
            timepoint = timepoint_dir.name
            file_map = {}
            for modality in MODALITIES:
                path = (
                    timepoint_dir
                    / f"{patient}_{timepoint}_{modality}.nii.gz"
                )
                if path.exists():
                    file_map[modality] = path
            mask_path = (
                timepoint_dir
                / f"{patient}_{timepoint}_MASK.nii.gz"
            )
            if mask_path.exists():
                file_map["MASK"] = mask_path
            scans.append(
                (
                    timepoint_dir,
                    patient,
                    timepoint,
                    file_map,
                )
            )
    return scans
def find_test_scans(data_dir: Path):
    """
    Test layout:
    test/
      P54/
        P54_T1.nii.gz
        P54_T2.nii.gz
        P54_FLAIR.nii.gz
        P54_MASK.nii.gz
    Each patient folder is one scan.
    """
    scans = []
    test_dir = data_dir / "test"
    if not test_dir.exists():
        return scans
    for patient_dir in sorted(test_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient = patient_dir.name
        file_map = {}
        for modality in MODALITIES:
            path = patient_dir / f"{patient}_{modality}.nii.gz"
            if path.exists():
                file_map[modality] = path
        mask_path = patient_dir / f"{patient}_MASK.nii.gz"
        if mask_path.exists():
            file_map["MASK"] = mask_path
        scans.append(
            (
                patient_dir,
                patient,
                "test",
                file_map,
            )
        )
    return scans
def main():
    parser = argparse.ArgumentParser(
        description="Verify MSLesSeg dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to extracted MSLesSeg dataset",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/processed/mslesseg_manifest.csv"
        ),
        help="Where to write verification manifest",
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir}"
        )
    # Discover scans
    training_scans = find_training_scans(data_dir)
    test_scans = find_test_scans(data_dir)
    all_scans = training_scans + test_scans
    print(
        f"Found {len(all_scans)} candidate patient-timepoint scans. "
        "Verifying..."
    )
    # Verify every scan
    rows = []
    for scan_dir, patient, timepoint, file_map in all_scans:
        row = check_scan(
            scan_dir,
            patient,
            timepoint,
            file_map,
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    # Statistics
    unique_patients = df["patient"].nunique()
    total_scans = len(df)
    passing_scans = int(df["ok"].sum())
    # Save manifest
    args.manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    df.to_csv(args.manifest, index=False)
    # Print summary
    print()
    print("=" * 70)
    print("MSLesSeg verification summary")
    print("=" * 70)
    print(f"Patients found:             {unique_patients}")
    print(f"Scans found:                {total_scans}")
    print(
        f"Scans passing all checks:   "
        f"{passing_scans} / {total_scans}"
    )
    print(
        f"Manifest written to:       "
        f"{args.manifest}"
    )
    print()
    # Expected counts
    if unique_patients != EXPECTED_PATIENTS:
        print(
            f"WARNING: expected {EXPECTED_PATIENTS} patients, "
            f"found {unique_patients}."
        )
    if total_scans != EXPECTED_SCANS:
        print(
            f"WARNING: expected {EXPECTED_SCANS} scans, "
            f"found {total_scans}."
        )
    if passing_scans != total_scans:
        print(
            "WARNING: some scans failed verification."
        )
        print()
        print("Problem scans:")
        problems = df[~df["ok"]]
        print(
            problems[
                [
                    "patient",
                    "timepoint",
                    "issues",
                ]
            ].to_string(index=False)
        )
    else:
        print(
            "ALL SCANS PASSED BASIC DATASET VERIFICATION."
        )
if __name__ == "__main__":
    main()