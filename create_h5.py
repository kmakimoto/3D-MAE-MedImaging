# Code to create .h5 file for COPDGene data to be used in the TANGERINE model
# Kalysta Makimoto 

"""
build_hdf5.py — Convert all .nrrd CT volumes into one shared HDF5 file,
with every CSV column (demographics, clinical variables, candidate labels)
attached as attributes on each volume's dataset. This decouples the
expensive resampling step from label selection — you can regenerate
train/val CSVs for any outcome later without touching this file again.
"""
import os
import h5py
import numpy as np
import pandas as pd
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# ---- Config ----------------------------------------------------------
TARGET_SIZE = (256, 256, 256)     # (D, H, W) — divisible by patch_size=16
HU_MIN, HU_MAX = -1200, 800
OUTPUT_H5 = "ct_volumes.h5"
LABELS_CSV = "your_labels.csv"    # TODO: path to your full clinical CSV
N_WORKERS = 8

ID_COLUMN = "cid"          # TODO: confirm — case ID
STUDY = "COPDGene"          # TODO: confirm / parameterize if multi-study
def cid_to_nrrd_path(cid):
    return f"/data/tmp/{STUDY}/nnunet/input/{cid}_0000.nrrd"


# ---- Worker: resample only, no HDF5 access (runs in subprocess) --------
def resample_to_fixed_size(sitk_image, target_size_dhw):
    original_size = sitk_image.GetSize()
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


def process_one(cid):
    nrrd_path = cid_to_nrrd_path(cid)
    if not os.path.exists(nrrd_path):
        return cid, None, f"missing file: {nrrd_path}"
    try:
        img = sitk.ReadImage(nrrd_path)
        img = resample_to_fixed_size(img, TARGET_SIZE)
        arr = sitk.GetArrayFromImage(img)
        arr = np.clip(arr, HU_MIN, HU_MAX).astype(np.int16)
        return cid, arr, None
    except Exception as e:
        return cid, None, str(e)


def attr_safe(value):
    """h5py attrs can't store NaN-as-object or arbitrary Python objects cleanly —
    coerce to a type it handles well."""
    if pd.isna(value):
        return "NA"          # store missingness explicitly, not silently
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


# ---- Main: parallel resample, serial write ------------------------------
def main():
    df = pd.read_csv(LABELS_CSV)
    assert ID_COLUMN in df.columns, f"CSV must have {ID_COLUMN!r}, found {df.columns.tolist()}"
    df[ID_COLUMN] = df[ID_COLUMN].astype(str)
    df = df.set_index(ID_COLUMN)

    cids = df.index.tolist()
    written_ids = []

    with h5py.File(OUTPUT_H5, "w") as h5f, \
         ProcessPoolExecutor(max_workers=N_WORKERS) as pool:

        futures = {pool.submit(process_one, cid): cid for cid in cids}

        for future in tqdm(as_completed(futures), total=len(futures)):
            cid, arr, err = future.result()
            if arr is None:
                print(f"WARNING: skipping {cid}: {err}")
                continue

            dset = h5f.create_dataset(cid, data=arr, compression="gzip", compression_opts=1)

            # Attach every CSV column as an attribute on this case's dataset
            row = df.loc[cid]
            for col, val in row.items():
                dset.attrs[col] = attr_safe(val)

            written_ids.append(cid)

    print(f"Wrote {len(written_ids)}/{len(cids)} volumes to {OUTPUT_H5}")
    print(f"Columns embedded per case: {df.columns.tolist()}")


if __name__ == "__main__":
    main()