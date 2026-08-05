"""
make_label_csv.py — Generate train.csv/val.csv for a chosen outcome column,
joining the clinical CSV against the HDF5's subject keys. Does not open or
decompress any volume data, so this runs in seconds regardless of dataset size.

Usage:
    python make_label_csv.py --outcome_column finalGold_P1
    python make_label_csv.py --outcome_column ATS_PackYears_P1 --train_csv train_py.csv --val_csv val_py.csv
"""
import argparse
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="/home/km2347/ct_volumes.h5")
    p.add_argument("--clinical_csv", default="/data/km2347/copdgene_labels.csv",
                    help="TODO: confirm actual path")
    p.add_argument("--sep", default="\t", help="clinical_csv delimiter — confirmed tab, not comma")
    p.add_argument("--id_column", default="sid")
    p.add_argument("--outcome_column", required=True)
    p.add_argument("--train_csv", default="train.csv")
    p.add_argument("--val_csv", default="val.csv")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # ---- Which subjects actually have a volume in the HDF5 ----
    with h5py.File(args.h5, "r") as f:
        available_sids = set(f.keys())
    print(f"{len(available_sids)} subjects available in {args.h5}")

    # ---- Load clinical data, tab-separated per the real file format ----
    clinical = pd.read_csv(args.clinical_csv, sep=args.sep, low_memory=False)
    if args.id_column not in clinical.columns:
        raise ValueError(f"'{args.id_column}' not in CSV columns: {clinical.columns.tolist()}")
    if args.outcome_column not in clinical.columns:
        raise ValueError(f"'{args.outcome_column}' not in CSV columns: {clinical.columns.tolist()}")

    clinical[args.id_column] = clinical[args.id_column].astype(str)
    clinical = clinical[clinical[args.id_column].isin(available_sids)]
    print(f"{len(clinical)} of those subjects found in {args.clinical_csv}")

    # ---- Missingness: treat both true NaN and whitespace-only strings as missing ----
    # (the CSV encodes missing values both ways, per finalGold_P2 investigation)
    outcome = clinical[args.outcome_column]
    is_missing = outcome.isna() | (
        outcome.astype(str).str.strip() == ""
    )
    clinical = clinical[~is_missing]
    print(f"{len(clinical)} subjects have a non-missing '{args.outcome_column}' label")

    if len(clinical) == 0:
        raise ValueError(f"No subjects have a non-missing '{args.outcome_column}' value — check the column name/values.")

    df = clinical[[args.id_column, args.outcome_column]].rename(
        columns={args.id_column: "Path", args.outcome_column: "Label"}
    )

    # ---- Split ----
    # sid IS the subject-level unit in this cohort (confirmed: filtered to one
    # scan per subject via INSP/STD/COPD), so a plain split on sid is already
    # patient-level correct — no separate subject-id derivation needed here.
    stratify = df["Label"] if df["Label"].nunique() <= 10 else None  # skip stratify for continuous outcomes
    train_df, val_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )

    train_df.to_csv(args.train_csv, index=False)
    val_df.to_csv(args.val_csv, index=False)
    print(f"Train: {len(train_df)} rows -> {args.train_csv}")
    print(f"Val:   {len(val_df)} rows -> {args.val_csv}")
    print(f"Label value counts (train):\n{train_df['Label'].value_counts()}")


if __name__ == "__main__":
    main()