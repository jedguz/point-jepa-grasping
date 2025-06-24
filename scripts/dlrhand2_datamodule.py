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
    Loads all grasps and scores from recording.npz files, with one mesh sample per file.
    - recording.npz has 'grasps': (N,19), 'scores': (N,)
    - Samples mesh only once per file and reuses for all grasps.
    Returns (pc, grasp_vec, score).
    """
    def __init__(self, root_dir: str, num_points: int = 2048):
        super().__init__()
        # Resolve project root if Hydra changed working dir
        try:
            base = get_original_cwd()
        except (ValueError, NameError):
            base = os.getcwd()
        root = os.path.join(base, root_dir)
        pattern = os.path.join(root, "**", "recording.npz")
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            raise FileNotFoundError(f"No recording.npz found under {root}")
        self.num_points = num_points
        self.samples = []        # list of (file, grasp_idx)
        self.pc_cache = {}       # map file -> sampled PC
        for f in files:
            data = np.load(f)
            grasps = data['grasps']        # (N,19)
            scores = data['scores']        # (N,)
            # Pre-sample mesh once
            mesh_file = os.path.join(os.path.dirname(f), 'mesh.obj')
            pc = sample_mesh_as_pc(mesh_file, self.num_points)
            self.pc_cache[f] = pc         # store (num_points,3)
            N = grasps.shape[0]
            for i in range(N):
                self.samples.append((f, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        f, gi = self.samples[idx]
        data = np.load(f)
        grasp_vec = data['grasps'][gi].astype(np.float32)
        score = float(data['scores'][gi])
        pc = self.pc_cache[f]        # reuse sampled PC
        return (
            torch.from_numpy(pc),               # (num_points,3)
            torch.from_numpy(grasp_vec),        # (19,)
            torch.tensor(score, dtype=torch.float32)
        )

class DLRHand2DataModule(pl.LightningDataModule):
    """
    LightningDataModule for DLR-Hand-2 grasps: loads pointclouds + grasps + scores.
    Configurable via Hydra: root_dir, batch_size, num_workers, num_points.
    """
    def __init__(
        self,
        root_dir: str = 'data/grasp_sample/03948459',
        batch_size: int = 4,
        num_workers: int = 0,
        num_points: int = 2048
    ):
        super().__init__()
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
