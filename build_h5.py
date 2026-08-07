# Code to create .h5 file for COPDGene data to be used in the TANGERINE model
# Kalysta Makimoto 

"""
build_h5.py — Convert COPDGene .nrrd CT volumes (Phase 1 / STD kernel / INSP)
into a single HDF5 file of fixed-size (256, 256, 256), HU-clipped, int16 cubes,
based on the TANGERINE paper. 

Run directly on `master`.

Resume-safe: if OUTPUT_H5 already exists, already-written sids are skipped,
so a rerun after a crash/interruption picks up where it left off.

Usage:
    # Small test run first 
    python build_h5.py --limit 10

    # Full run cohort
    python build_h5.py
"""
import os
import glob
import argparse
import h5py
import numpy as np
import pandas as pd
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ---- Config -------------------------------------------------------------
RAW_ROOT = "/copd/Processed/COPDGene"          # source, local to master
OUTPUT_H5 = "/home/km2347/ct_volumes.h5"       # local
LABELS_CSV = "/home/km2347/COPDGene_Data/COPDGene_P1P2P3_Flat_SM_NS_Sep24.txt" 

TARGET_SIZE = (256, 256, 256)   # (D, H, W) — matches TANGERINE pretraining resolution
HU_MIN, HU_MAX = -1200, 800
N_WORKERS = 4                    

ID_COLUMN = "sid"     

DEMO_COLUMNS = ["gender", "Age_P1", "race", "ATS_PackYears_P1", 
                "smoking_status_P1", "finalGold_P1", 
                "finalGold_P2", "binary_FEV1_decline_60",
                "multiclass_FEV1_decline"] 

MISSING_LOG = "missing_or_ambiguous_sids.txt"


# ---- Path resolution: Phase 1 (COPD), STD kernel, INSP only ------------
def sid_to_nrrd_path(sid):
    """
    Naming convention: {sid}_INSP_STD_{site}_COPD/{sid}_INSP_STD_{site}_COPD.nrrd
    """
    pattern = os.path.join(RAW_ROOT, sid, f"{sid}_INSP_STD_*_COPD", f"{sid}_INSP_STD_*_COPD.nrrd")
    matches = glob.glob(pattern)
    if len(matches) == 0:
        return None, "no_match"
    if len(matches) > 1:
        return None, f"ambiguous:{matches}"
    return matches[0], None


# ---- Worker: resample only, no HDF5 access (runs in subprocess) --------
def resample_to_fixed_size(sitk_image, target_size_dhw):
    original_size = sitk_image.GetSize()          # SimpleITK order: (W, H, D)
    original_spacing = sitk_image.GetSpacing()
    target_size_sitk = (target_size_dhw[2], target_size_dhw[1], target_size_dhw[0])
    new_spacing = [
        osz * osp / nsz
        for osz, osp, nsz in zip(original_size, original_spacing, target_size_sitk)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(target_size_sitk)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetInterpolator(sitk.sitkBSpline)
    return resample.Execute(sitk_image)


def process_one(sid):
    """Runs in a worker process. Returns (sid, array_or_None, cid_or_None, error_or_None)."""
    nrrd_path, path_err = sid_to_nrrd_path(sid)
    if nrrd_path is None:
        return sid, None, None, path_err

    try:
        img = sitk.ReadImage(nrrd_path)
        img = resample_to_fixed_size(img, TARGET_SIZE)
        arr = sitk.GetArrayFromImage(img)                # (D, H, W)
        arr = np.clip(arr, HU_MIN, HU_MAX).astype(np.int16)
        cid = os.path.basename(nrrd_path).replace(".nrrd", "")
        return sid, arr, cid, None
    except Exception as e:
        return sid, None, None, str(e)


# ---- Main -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N subjects — use for a test run before the full 10k job")
    args = parser.parse_args()
 
    df = pd.read_csv(LABELS_CSV, sep="\t")
    assert ID_COLUMN in df.columns, f"CSV must have column {ID_COLUMN!r}, found {df.columns.tolist()}"
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
 
    missing_demo_cols = [c for c in DEMO_COLUMNS if c not in df.columns]
    if missing_demo_cols:
        raise ValueError(f"DEMO_COLUMNS not found in CSV: {missing_demo_cols}. "
                          f"Available columns: {df.columns.tolist()}")
 
    df_by_sid = df.set_index(ID_COLUMN)   # for fast per-sid attr lookup in the write loop
    sids = df[ID_COLUMN].tolist()
 
    if args.limit:
        sids = sids[:args.limit]
        print(f"TEST RUN: limited to first {len(sids)} subjects")
 
    mode = "a" if os.path.exists(OUTPUT_H5) else "w"
    if mode == "a":
        print(f"{OUTPUT_H5} already exists — resuming, will skip already-written sids")
 
    written = []
    skipped = []
 
    os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)
 
    with h5py.File(OUTPUT_H5, mode) as h5f, \
         ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
 
        already_done = set(h5f.keys())
        sids_to_process = [s for s in sids if s not in already_done]
        print(f"{len(already_done)} already in {OUTPUT_H5}, processing {len(sids_to_process)} remaining")
 
        futures = {pool.submit(process_one, sid): sid for sid in sids_to_process}
 
        for future in tqdm(as_completed(futures), total=len(futures)):
            sid, arr, cid, err = future.result()
            if arr is None:
                skipped.append((sid, err))
                continue
 
            dset = h5f.create_dataset(sid, data=arr, compression="gzip", compression_opts=1)
            dset.attrs["sid"] = sid
            dset.attrs["cid"] = cid
 
            if sid in df_by_sid.index:
                row = df_by_sid.loc[sid]
                for col in DEMO_COLUMNS:
                    val = row[col]
                    if pd.isna(val):
                        dset.attrs[col] = "NA"
                    elif isinstance(val, (int, float, np.integer, np.floating)):
                        dset.attrs[col] = float(val)
                    else:
                        dset.attrs[col] = str(val)
 
            written.append(sid)
 
    print(f"\nWrote {len(written)} volumes to {OUTPUT_H5}")
    print(f"Skipped {len(skipped)} subjects (missing or ambiguous matches)")
 
    if skipped:
        with open(MISSING_LOG, "w") as f:
            for sid, reason in skipped:
                f.write(f"{sid}\t{reason}\n")
        print(f"Details written to {MISSING_LOG} — review before the full run" if args.limit
              else f"Details written to {MISSING_LOG}")
 
 
if __name__ == "__main__":
    main()