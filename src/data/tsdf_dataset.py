import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TSDFVoxelDataset(Dataset):
    """
    A simple Dataset that loads 3D TSDF volumes from .npy files
    and returns a dict {'points': tensor([1, D, H, W])} for Voxel-MAE.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.file_list = sorted([f for f in os.listdir(root_dir) if f.endswith('.npy')])
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.file_list[idx])
        tsdf = np.load(path)                # shape (D, H, W)
        tsdf = torch.from_numpy(tsdf).float().unsqueeze(0)  # (1, D, H, W)
        sample = {'points': tsdf}
        if self.transform:
            sample = self.transform(sample)
        return sample