"""
test_dataset.py — Sanity check Custom3DDataset reads correctly from ct_volumes.h5
"""
import pandas as pd
import h5py

H5_PATH = "/home/km2347/ct_volumes.h5"

with h5py.File(H5_PATH, "r") as f:
    sids = list(f.keys())

print(f"Building smoke-test CSV from {len(sids)} real subjects: {sids}")

# Dummy binary labels — only to exercise the pipeline, not real training data
test_df = pd.DataFrame({
    "Path": sids,
    "Label": [i % 2 for i in range(len(sids))],
})
test_df.to_csv("test_smoke.csv", index=False)
print(test_df)

# ---- Now exercise Custom3DDataset itself ----
from datasets_three_d_fine_h5 import Custom3DDataset

ds = Custom3DDataset(csv_path="test_smoke.csv", h5_path=H5_PATH)
print(f"\nDataset length: {len(ds)}")

for i in range(len(ds)):
    vol, label = ds[i]
    print(f"[{i}] sid={sids[i]}  vol shape={tuple(vol.shape)}  dtype={vol.dtype}  "
          f"min={vol.min():.4f}  max={vol.max():.4f}  label={label.item()}")