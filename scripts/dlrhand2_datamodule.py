# scripts/dlrhand2_datamodule.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import trimesh
from hydra.utils import get_original_cwd

def sample_mesh_as_pc(mesh_path: str, num_points: int = 2048) -> np.ndarray:
    """
    Load a mesh using trimesh and uniformly sample num_points points on its surface.
    """
    mesh = trimesh.load(mesh_path, process=False)
    if not mesh.is_watertight:
        mesh = mesh.convex_hull
    pts, _ = mesh.sample(num_points, return_index=True)
    return pts.astype(np.float32)

class DLRHand2Dataset(Dataset):
    """
    Finds any 'recording.npz' under root_dir (recursively), and expects alongside
    each a mesh.obj file in the same folder.
    Returns pointcloud and dummy score for testing.
    """
    def __init__(self, root_dir: str, num_points: int = 2048):
        super().__init__()
        # Ensure we search from original project root when Hydra has changed CWD
        base_dir = get_original_cwd()
        self.root_dir = os.path.join(base_dir, root_dir)
        pattern = os.path.join(self.root_dir, "**", "recording.npz")
        self.files = sorted(glob.glob(pattern, recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No recording.npz found under {self.root_dir}")
        self.num_points = num_points

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        npz_path = self.files[idx]
        data = np.load(npz_path)
        # The npz may already contain a pointcloud, else sample mesh
        key = data.files[0]
        pc = data[key].astype(np.float32)
        if pc.ndim != 2 or pc.shape[1] != 3:
            # fallback to sampling mesh
            mesh_file = os.path.join(os.path.dirname(npz_path), 'mesh.obj')
            pc = sample_mesh_as_pc(mesh_file, self.num_points)
        else:
            # optionally down-/up-sample to num_points
            if pc.shape[0] != self.num_points:
                idxs = np.random.choice(pc.shape[0], self.num_points,
                                       replace=(pc.shape[0] < self.num_points))
                pc = pc[idxs]
        return torch.from_numpy(pc), torch.tensor(0.0)

class DLRHand2DataModule(pl.LightningDataModule):
    """
    root_dir: base folder to search for recording.npz
    num_points: points per cloud
    batch_size, num_workers configurable
    """
    def __init__(
        self,
        root_dir: str = 'data/grasp_sample/03948459',
        batch_size: int = 4,
        num_workers: int = 0,
        num_points: int = 2048
    ):
        super().__init__()
        # Store the relative path; actual resolution happens in Dataset
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_points = num_points

    def setup(self, stage=None):
        self.dataset = DLRHand2Dataset(
            root_dir=self.root_dir,
            num_points=self.num_points
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return self.train_dataloader()

    def test_dataloader(self):
        return self.train_dataloader()