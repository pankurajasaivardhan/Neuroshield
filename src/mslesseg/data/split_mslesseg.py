from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
RANDOM_SEED = 42
def main():
    parser = argparse.ArgumentParser(
        description="Create patient-level train/validation splits for MSLesSeg."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/mslesseg_manifest.csv"),
        help="Path to verified MSLesSeg manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/splits"),
        help="Directory for split CSV files.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.20,
        help="Fraction of training patients used for validation.",
    )
    args = parser.parse_args()
    # Load verified manifest
    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {args.manifest}"
        )
    df = pd.read_csv(args.manifest)
    # Only verified scans
    df = df[df["ok"] == True].copy()
    if df.empty:
        raise RuntimeError("No verified scans found in manifest.")
    # Identify patients by original dataset split
    train_scans = df[df["timepoint"] != "test"].copy()
    test_scans = df[df["timepoint"] == "test"].copy()
    train_patients = sorted(train_scans["patient"].unique())
    test_patients = sorted(test_scans["patient"].unique())
    print("=" * 70)
    print("MSLesSeg patient-level split")
    print("=" * 70)
    print(f"Training patients available : {len(train_patients)}")
    print(f"Official test patients      : {len(test_patients)}")
    print()
    # Split ONLY the training patients
    train_patients_final, val_patients = train_test_split(
        train_patients,
        test_size=args.val_size,
        random_state=RANDOM_SEED,
    )
    train_patients_final = sorted(train_patients_final)
    val_patients = sorted(val_patients)
    # Create split tables
    train_df = train_scans[
        train_scans["patient"].isin(train_patients_final)
    ].copy()
    val_df = train_scans[
        train_scans["patient"].isin(val_patients)
    ].copy()
    test_df = test_scans.copy()
    # Add explicit split labels
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"
    # Save
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    train_path = args.output_dir / "train.csv"
    val_path = args.output_dir / "validation.csv"
    test_path = args.output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    # Combined file
    combined_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True,
    )
    combined_path = args.output_dir / "splits.csv"
    combined_df.to_csv(combined_path, index=False)
    # Print summary
    print("Patient split:")
    print(f"  Train      : {len(train_patients_final)} patients")
    print(f"  Validation : {len(val_patients)} patients")
    print(f"  Test       : {len(test_patients)} patients")
    print()
    print("Scan/timepoint split:")
    print(f"  Train      : {len(train_df)} scans")
    print(f"  Validation : {len(val_df)} scans")
    print(f"  Test       : {len(test_df)} scans")
    print()
    print("Files written:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {combined_path}")
    print()
    # Safety checks
    train_set = set(train_patients_final)
    val_set = set(val_patients)
    test_set = set(test_patients)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert (
        len(train_set | val_set | test_set)
        == len(train_patients) + len(test_patients)
    )
    print("Patient leakage check: PASSED")
    print("No patient appears in more than one split.")
if __name__ == "__main__":
    main()