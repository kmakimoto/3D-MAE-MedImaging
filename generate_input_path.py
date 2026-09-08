#!/usr/bin/env python3
"""Generate the input CSV for extract_encoder_embeddings.py by resolving each SID from
a CSV of subject IDs to its NRRD file path, using the Phase 1 (COPD), STD-kernel,
INSP-only naming convention:

    {RAW_ROOT}/{sid}/{sid}_INSP_STD_{site}_COPD/{sid}_INSP_STD_{site}_COPD.nrrd

The output CSV contains exactly two columns: Path and Label. The Label value for
each subject is pulled from --label_col in --sid_csv.
"""

import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Path/Label CSV by resolving each SID in --sid_csv to its "
                     "NRRD file path under --raw_root, and pulling the label value from "
                     "--label_col in the same sheet."
    )
    parser.add_argument(
        "--sid_csv", required=True, type=str,
        help="CSV (or Excel file) containing the list of subject IDs plus any other columns.",
    )
    parser.add_argument(
        "--sid_col", default="sid", type=str,
        help="Column name in --sid_csv holding subject IDs (default: sid).",
    )
    parser.add_argument(
        "--label_col", required=True, type=str,
        help="Column name in --sid_csv holding the value to use as 'Label' in the output CSV.",
    )
    parser.add_argument(
        "--raw_root", default="/datagpu/datasets/km2347/COPDGene_master", type=str,
        help="Root directory containing per-subject folders (RAW_ROOT).",
    )
    parser.add_argument(
        "--output_csv", default='/home/km2347/3D-MAE-MedImaging/example_inputs/input.csv', type=str,
        help="Where to write the resolved Path/Label CSV. This is what you pass as "
             "--input_csv to extract_encoder_embeddings.py.",
    )
    parser.add_argument(
        "--failures_csv", default='/home/km2347/3D-MAE-MedImaging/example_inputs/failures.csv', type=str,
        help="Optional path to also write a CSV of SIDs that failed to resolve, with a reason.",
    )
    parser.add_argument(
        "--expected_count", default=None, type=int,
        help="If given, warn when the number of successfully resolved subjects doesn't match this.",
    )
    return parser


def sid_to_nrrd_path(sid: str, raw_root: str):
    """
    Naming convention: {sid}_INSP_STD_{site}_COPD/{sid}_INSP_STD_{site}_COPD.nrrd
    """
    pattern = os.path.join(raw_root, sid, f"{sid}_INSP_STD_*_COPD", f"{sid}_INSP_STD_*_COPD.nrrd")
    matches = glob.glob(pattern)
    if len(matches) == 0:
        return None, "no_match"
    if len(matches) > 1:
        return None, f"ambiguous:{matches}"
    return matches[0], None


def read_sheet(path: str) -> pd.DataFrame:
    """Reads a CSV or Excel file based on its extension."""
    ext = Path(path).suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def main() -> None:
    args = get_args_parser().parse_args()

    sid_df = read_sheet(args.sid_csv)

    missing_cols = {args.sid_col, args.label_col} - set(sid_df.columns)
    if missing_cols:
        raise ValueError(
            f"Column(s) {missing_cols} not found in {args.sid_csv}. "
            f"Available columns: {sid_df.columns.tolist()}"
        )

    # Keep sid -> label lookup so we can attach the label after path resolution.
    sid_df["__sid_str"] = sid_df[args.sid_col].astype(str)
    label_lookup = sid_df.set_index("__sid_str")[args.label_col]

    sids = sid_df["__sid_str"].tolist()

    resolved_rows = []
    failed_rows = []

    for sid in sids:
        path, error = sid_to_nrrd_path(sid, args.raw_root)
        if error is None:
            resolved_rows.append({"Path": path, "Label": label_lookup.loc[sid]})
        else:
            failed_rows.append({"SID": sid, "Reason": error})

    resolved_df = pd.DataFrame(resolved_rows, columns=["Path", "Label"])

    dup_paths = resolved_df["Path"][resolved_df["Path"].duplicated()].tolist()
    if dup_paths:
        print(f"WARNING: duplicate resolved paths: {dup_paths}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_df.to_csv(output_path, index=False)

    print(f"Resolved {len(resolved_df)} / {len(sids)} SIDs to NRRD paths.")
    print(f"Wrote CSV to {output_path}")

    if failed_rows:
        print(f"WARNING: {len(failed_rows)} SID(s) failed to resolve:")
        for row in failed_rows[:20]:
            print(f"  - {row['SID']}: {row['Reason']}")
        if len(failed_rows) > 20:
            print(f"  ... ({len(failed_rows) - 20} more)")

        if args.failures_csv:
            failures_path = Path(args.failures_csv)
            failures_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(failed_rows).to_csv(failures_path, index=False)
            print(f"Wrote failures to {failures_path}")

    if args.expected_count is not None and len(resolved_df) != args.expected_count:
        print(
            f"WARNING: expected {args.expected_count} resolved subjects but got "
            f"{len(resolved_df)}. Check --raw_root, --sid_csv, and any SIDs listed "
            f"as no_match/ambiguous above."
        )


if __name__ == "__main__":
    main()