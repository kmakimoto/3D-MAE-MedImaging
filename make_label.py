"""
make_label_csv.py — Generate train.csv/val.csv/test.csv for a chosen binary
outcome column (e.g. binary_FEV1_decline_60, or finalGold_P1 binarized as
no COPD (healthy, PRISm, GOLD 0) vs COPD).

Joins the clinical CSV against the HDF5's subject keys only — no volume data
is opened, so this runs in seconds regardless of dataset size. Rerun any time
you want a different split/seed/label; nothing here touches ct_volumes.h5.

Usage:
    python make_label_csv.py --clinical_csv /path/to/real_labels.txt \
        --label_column binary_FEV1_decline_60

    python make_label_csv.py --clinical_csv /path/to/real_labels.txt \
        --label_column finalGold_P1
"""
import argparse
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# GOLD stage -> binary label, explicit codes only. Only used when
# --label_column finalGold_P1 is selected.
# -2, -1, 0 -> no COPD (0); 1, 2, 3, 4 -> any COPD (1).
# Anything not listed here is dropped, not guessed.
GOLD_TO_BINARY = {
    "-2": 0,
    "-1": 0,
    "0": 0,
    "1": 1,
    "2": 1,
    "3": 1,
    "4": 1,
}

# Values accepted as already-binary (e.g. binary_FEV1_decline_60). Anything
# not listed here is dropped, not guessed.
GENERIC_BINARY_MAP = {
    "0": 0,
    "0.0": 0,
    "1": 1,
    "1.0": 1,
}


def binarize_gold(raw_value):
    """GOLD-stage -> {0, 1, None}. None means: drop this subject."""
    if pd.isna(raw_value):
        return None
    s = str(raw_value).strip()
    if s == "":
        return None
    return GOLD_TO_BINARY.get(s, None)


def binarize_generic(raw_value):
    """Already-binary column -> {0, 1, None}. None means: drop this subject."""
    if pd.isna(raw_value):
        return None
    s = str(raw_value).strip()
    if s == "":
        return None
    return GENERIC_BINARY_MAP.get(s, None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="/home/km2347/ct_volumes.h5")
    p.add_argument("--clinical_csv", required=True, help="path to the real tab-separated labels file")
    p.add_argument("--sep", default="\t")
    p.add_argument("--id_column", default="sid")
    p.add_argument("--label_column", default="binary_FEV1_decline_60",
                    help="clinical column to use as the outcome (e.g. binary_FEV1_decline_60, finalGold_P1)")
    p.add_argument("--train_csv", default="train.csv")
    p.add_argument("--val_csv", default="val.csv")
    p.add_argument("--test_csv", default="test.csv")
    p.add_argument("--train_frac", type=float, default=0.50)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--test_frac", type=float, default=0.40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with h5py.File(args.h5, "r") as f:
        available_sids = set(f.keys())
    print(f"{len(available_sids)} subjects available in {args.h5}")

    clinical = pd.read_csv(args.clinical_csv, sep=args.sep, low_memory=False)
    if args.id_column not in clinical.columns:
        raise ValueError(f"'{args.id_column}' not in CSV columns: {clinical.columns.tolist()}")
    if args.label_column not in clinical.columns:
        raise ValueError(f"'{args.label_column}' not in CSV columns: {clinical.columns.tolist()}")

    clinical[args.id_column] = clinical[args.id_column].astype(str)
    clinical = clinical[clinical[args.id_column].isin(available_sids)]
    print(f"{len(clinical)} of those subjects found in {args.clinical_csv}")

    # GOLD stage needs the multi-value -> binary mapping; every other column
    # (e.g. binary_FEV1_decline_60) is treated as already-binary.
    if args.label_column == "finalGold_P1":
        binarizer = binarize_gold
    else:
        binarizer = binarize_generic
    clinical["Label"] = clinical[args.label_column].apply(binarizer)

    n_before = len(clinical)
    dropped_codes = clinical.loc[clinical["Label"].isna(), args.label_column].value_counts()
    clinical = clinical.dropna(subset=["Label"])
    clinical["Label"] = clinical["Label"].astype(int)
    print(f"{len(clinical)}/{n_before} subjects have a usable {args.label_column} value")
    if len(dropped_codes) > 0:
        print(f"Dropped due to missing/unrecognized {args.label_column} codes:\n{dropped_codes}")

    if len(clinical) == 0:
        raise ValueError(
            f"No subjects left after filtering {args.label_column} — "
            f"check the binarization mapping against real values."
        )

    df = clinical[[args.id_column, "Label"]].rename(columns={args.id_column: "Path"})

    frac_sum = args.train_frac + args.val_frac + args.test_frac
    if not np.isclose(frac_sum, 1.0):
        raise ValueError(f"train_frac + val_frac + test_frac must sum to 1.0, got {frac_sum}")

    # Two-step split: first carve off test set, then split the remainder into train/val.
    # Both steps stratify on Label so class balance is preserved in all three sets.
    train_val_df, test_df = train_test_split(
        df, test_size=args.test_frac, random_state=args.seed, stratify=df["Label"]
    )
    # val_frac as a fraction of the remaining (train+val) pool, not of the original total
    val_frac_of_remainder = args.val_frac / (args.train_frac + args.val_frac)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_frac_of_remainder, random_state=args.seed,
        stratify=train_val_df["Label"]
    )

    train_df.to_csv(args.train_csv, index=False)
    val_df.to_csv(args.val_csv, index=False)
    test_df.to_csv(args.test_csv, index=False)
    print(f"\nTrain: {len(train_df)} rows ({len(train_df)/len(df):.1%}) -> {args.train_csv}")
    print(f"Val:   {len(val_df)} rows ({len(val_df)/len(df):.1%}) -> {args.val_csv}")
    print(f"Test:  {len(test_df)} rows ({len(test_df)/len(df):.1%}) -> {args.test_csv}")
    print(f"\nLabel counts (train) — 0=negative, 1=positive:\n{train_df['Label'].value_counts()}")
    print(f"\nLabel counts (val):\n{val_df['Label'].value_counts()}")
    print(f"\nLabel counts (test):\n{test_df['Label'].value_counts()}")


if __name__ == "__main__":
    main()