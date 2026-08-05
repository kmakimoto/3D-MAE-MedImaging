"""
test_dataloader.py — Confirm Custom3DDataset works correctly through a
DataLoader with num_workers > 0 (exercises h5py fork-safety).
"""
import torch
from torch.utils.data import DataLoader
from datasets_three_d_fine_h5 import Custom3DDataset

H5_PATH = "/home/km2347/ct_volumes.h5"

ds = Custom3DDataset(csv_path="test_smoke.csv", h5_path=H5_PATH)

# num_workers=2 specifically to test the multiprocessing path, not just num_workers=0
loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=2)

for i, (batch_vols, batch_labels) in enumerate(loader):
    print(f"Batch {i}: vols={tuple(batch_vols.shape)}  dtype={batch_vols.dtype}  "
          f"min={batch_vols.min():.4f}  max={batch_vols.max():.4f}  labels={batch_labels.tolist()}")

print("\nDataLoader test passed — no crashes or hangs with num_workers=2")