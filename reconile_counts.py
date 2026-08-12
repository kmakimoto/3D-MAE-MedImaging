import h5py
import pandas as pd

H5_PATH = "/home/km2347/ct_volumes.h5"
CSV_PATH = "/home/km2347/COPDGene_Data/COPDGene_P1P2P3_Flat_SM_NS_Sep24.txt"
MISSING_LOG = "/home/km2347/3D-MAE-MedImaging/missing_or_ambiguous_sids.txt"

# --- Load each source as a set of sids ---
df = pd.read_csv(CSV_PATH, sep="\t", low_memory=False)
csv_sids_list = df["sid"].astype(str).tolist()
csv_sids = set(csv_sids_list)
print(f"CSV rows: {len(csv_sids_list)}, unique sids: {len(csv_sids)}")
if len(csv_sids_list) != len(csv_sids):
    print(f"  -> {len(csv_sids_list) - len(csv_sids)} DUPLICATE sid rows in the CSV")

with h5py.File(H5_PATH, "r") as f:
    written_sids = set(f.keys())
print(f"Written (in .h5): {len(written_sids)}")

skipped_sids = set()
with open(MISSING_LOG) as f:
    for line in f:
        sid = line.split("\t")[0].strip()
        if sid:
            skipped_sids.add(sid)
print(f"Skipped (in missing log): {len(skipped_sids)}")

# --- Sanity checks ---
overlap = written_sids & skipped_sids
print(f"\nOverlap between written AND skipped (should be 0): {len(overlap)}")
if overlap:
    print(f"  Example overlapping sids: {list(overlap)[:10]}")

written_not_in_csv = written_sids - csv_sids
print(f"Written sids NOT in CSV (should be 0): {len(written_not_in_csv)}")

skipped_not_in_csv = skipped_sids - csv_sids
print(f"Skipped sids NOT in CSV (should be 0): {len(skipped_not_in_csv)}")

accounted_for = written_sids | skipped_sids
never_attempted = csv_sids - accounted_for
print(f"\nCSV sids never written AND never in skip log: {len(never_attempted)}")
if never_attempted:
    print(f"  Example: {list(never_attempted)[:10]}")

print(f"\n--- Summary ---")
print(f"CSV unique sids:      {len(csv_sids)}")
print(f"Written:               {len(written_sids)}")
print(f"Skipped:                {len(skipped_sids)}")
print(f"Written + Skipped:      {len(written_sids) + len(skipped_sids)}")
print(f"Written | Skipped (dedup): {len(accounted_for)}")
print(f"Unaccounted for:        {len(never_attempted)}")