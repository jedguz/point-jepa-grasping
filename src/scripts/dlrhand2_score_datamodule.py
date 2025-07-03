"""
Lightning DataModule and PyTorch Dataset for DLR-Hand-2 grasp recordings.
Optimized for SSD-backed batch-by-batch loading.
"""

from __future__ import annotations
import os
import glob
from typing import List, Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from hydra.utils import get_original_cwd
import trimesh
import shutil

# ────────────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────────────
def sample_mesh_as_pc(mesh_path: str, num_points: int = 2048) -> np.ndarray:
    mesh = trimesh.load(mesh_path, process=False)
    if not mesh.is_watertight:
        mesh = mesh.convex_hull
    pts, _ = mesh.sample(num_points, return_index=True)
    return pts.astype(np.float32)

# ────────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────────
class DLRHand2Dataset(Dataset):
    def __init__(self, root_dir: str, num_points: int = 2048, ssd_cache_dir: str = "/mnt/disks/ssd/cache") -> None:
        super().__init__()
        self.source_root = os.path.abspath(root_dir)
        self.ssd_cache_root = os.path.abspath(ssd_cache_dir)
        os.makedirs(self.ssd_cache_root, exist_ok=True)
        self.num_points = num_points

        # Collect all files
        pattern = os.path.join(self.source_root, "**", "recording.npz")
        self.recording_files = sorted(glob.glob(pattern, recursive=True))
        if not self.recording_files:
            raise FileNotFoundError(f"No recordings found under {self.source_root}")

        self.samples: List[Tuple[str, int]] = []
        self.pc_cache = {}
        self.data_cache = {}

        for f in self.recording_files:
            num_grasps = int(np.load(f)["grasps"].shape[0])
            for gi in range(num_grasps):
                self.samples.append((f, gi))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        file_path, grasp_idx = self.samples[idx]

        # Cache recording file + mesh.obj to SSD
        cached_npz = self._ensure_cached(file_path)
        mesh_path = os.path.join(os.path.dirname(file_path), "mesh.obj")
        cached_mesh = self._ensure_cached(mesh_path)

        # Load into memory
        if cached_npz not in self.data_cache:
            self.data_cache[cached_npz] = np.load(cached_npz)
        data = self.data_cache[cached_npz]

        if cached_npz not in self.pc_cache:
            self.pc_cache[cached_npz] = sample_mesh_as_pc(cached_mesh, self.num_points)

        pc = self.pc_cache[cached_npz]
        grasp_vec = data["grasps"][grasp_idx].astype(np.float32)
        score = float(data["scores"][grasp_idx])

        return (
            torch.from_numpy(pc),
            torch.from_numpy(grasp_vec),
            torch.tensor(score, dtype=torch.float32),
        )

    def _ensure_cached(self, src_path: str) -> str:
        # Relativize to source root and mirror in SSD
        rel_path = os.path.relpath(src_path, self.source_root)
        dst_path = os.path.join(self.ssd_cache_root, rel_path)
        if not os.path.exists(dst_path):
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
        return dst_path

# ────────────────────────────────────────────────────────────────────────────────
# DataModule
# ────────────────────────────────────────────────────────────────────────────────
class DLRHand2DataModule(pl.LightningDataModule):
    """
    SSD-optimized Lightning DataModule for DLR-Hand-2 dataset.
    Does not preload into RAM — all samples loaded from disk per batch.
    """

    def __init__(
        self,
        root_dir: str = "data/grasp_sample/03948459/",
        ssd_cache_dir: str = "/mnt/disks/ssd/dataset",
        batch_size: int = 8,
        num_workers: int = 2,
        num_points: int = 2048,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset: DLRHand2Dataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.dataset is None:
            dataset_root = os.path.join(get_original_cwd(), self.hparams.root_dir)
            self.dataset = DLRHand2Dataset(
                root_dir=dataset_root,
                num_points=self.hparams.num_points,
                ssd_cache_dir="/mnt/disks/ssd/cache"
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
