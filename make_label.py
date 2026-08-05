"""
make_label_csv.py — Generate train.csv/val.csv for a chosen outcome column,
reading only HDF5 attributes (no volume decompression, no reprocessing).
"""
import argparse
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True)
    p.add_argument("--outcome_column", required=True)
    p.add_argument("--subject_column", default=None,
                    help="patient ID attr for split; if omitted, derives from cid prefix")
    p.add_argument("--train_csv", default="train.csv")
    p.add_argument("--val_csv", default="val.csv")
    p.add_argument("--test_size", type=float, default=0.2)
    args = p.parse_args()

    rows = []
    with h5py.File(args.h5, "r") as f:
        for cid in f.keys():
            attrs = dict(f[cid].attrs)
            if args.outcome_column not in attrs or attrs[args.outcome_column] == "NA":
                continue  # skip cases missing this particular label
            sid = attrs[args.subject_column] if args.subject_column else cid.split("_")[0]
            rows.append({"Path": cid, "Label": attrs[args.outcome_column], "sid": sid})

    df = pd.DataFrame(rows)
    print(f"{len(df)} cases have a non-missing '{args.outcome_column}' label")

    subjects = df["sid"].unique()
    subj_labels = df.drop_duplicates("sid").set_index("sid")["Label"]
    train_subj, val_subj = train_test_split(
        subjects, test_size=args.test_size, random_state=42,
        stratify=subj_labels.loc[subjects]
    )

    df[df["sid"].isin(train_subj)][["Path", "Label"]].to_csv(args.train_csv, index=False)
    df[df["sid"].isin(val_subj)][["Path", "Label"]].to_csv(args.val_csv, index=False)
    print(f"Wrote {args.train_csv}, {args.val_csv}")

if __name__ == "__main__":
    main()