# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import os
from typing import Tuple, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py


class Custom3DDataset(Dataset):
    """
    Loads 3D CT volumes from a shared HDF5 file, with labels supplied via CSV.

    The CSV must contain:
      - 'Path': the HDF5 dataset key for this case (i.e. a subject 'sid')
      - 'Label': the corresponding label

    Volumes are stored pre-resampled and HU-clipped as int16 in the HDF5 file
    (see build_h5.py). This class only normalizes to [0, 1] and adds the
    channel dimension — no SimpleITK/resampling work happens here.
    """

    def __init__(self, csv_path: str, h5_path: str, transform: Optional[callable] = None) -> None:
        self.h5_path = h5_path
        self.transform = transform
        self.samples = self._load_samples(csv_path)

        # Same HU range used when the volumes were written in build_h5.py —
        # keep these in sync if that ever changes.
        self.hu_min = -1200
        self.hu_max = 800

    def _load_samples(self, csv_path: str) -> pd.DataFrame:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        data = pd.read_csv(csv_path)
        required_columns = {'Path', 'Label'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"CSV file must contain columns {required_columns}. Found: {data.columns.tolist()}")
        data['Path'] = data['Path'].astype(str)  # ensure sid matches HDF5 key type exactly
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sid = self.samples.iloc[idx]['Path']
        label = self.samples.iloc[idx]['Label']

        # Open fresh per call — do NOT hold this open across DataLoader workers,
        # h5py file handles are not fork/pickle-safe.
        with h5py.File(self.h5_path, "r") as f:
            if sid not in f:
                raise KeyError(f"'{sid}' not found in {self.h5_path}")
            volume = f[sid][:].astype(np.float32)

        # Min-max normalization to [0, 1]
        volume = (volume + 1200) / (800 + 1200)
        volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]

        if self.transform:
            volume_tensor = self.transform(volume_tensor)

        return volume_tensor, torch.tensor(label, dtype=torch.float32)