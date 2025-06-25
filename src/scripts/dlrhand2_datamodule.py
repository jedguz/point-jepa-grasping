# scripts/dlrhand2_datamodule.py
"""
Lightning DataModule and PyTorch Dataset for DLR-Hand-2 grasp recordings.

Main improvements over the previous version
-------------------------------------------
1. **`data_cache`** – keeps every `recording.npz` in memory after first load,
   eliminating repeated disk I/O inside `__getitem__`.
2. **Deterministic validation / test loaders** – they no longer reuse the
   shuffling train loader.
3. Nit: small refactor around `pin_memory`.

The module still uses the *same* underlying dataset for train/val/test; only
the shuffling flag differs.  Add an explicit split if/when you have one.
"""
from __future__ import annotations

import glob
import shutil
import os
from typing import Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader, Dataset
import trimesh


# ────────────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────────────
def sample_mesh_as_pc(mesh_path: str, num_points: int = 2_048) -> np.ndarray:
    """
    Load a mesh and sample `num_points` points uniformly from its surface.

    If the mesh is not watertight we fall back to its convex hull to avoid
    strange behaviour in trimesh’s sampling routine.
    """
    mesh = trimesh.load(mesh_path, process=False)
    if not mesh.is_watertight:
        mesh = mesh.convex_hull
    pts, _ = mesh.sample(num_points, return_index=True)
    return pts.astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────────
class DLRHand2Dataset(Dataset):
    """
    Each item returns `(point_cloud, grasp_vec, score)`:

    * **point_cloud** – (N, 3) float32
    * **grasp_vec**   – (19,)  float32
    * **score**       – ()     float32 scalar
    """

    def __init__(self, root_dir: str, num_points: int = 2048) -> None:
        super().__init__()

        # Resolve original working directory when running under Hydra
        try:
            base = get_original_cwd()
        except (ValueError, NameError):
            base = os.getcwd()
        root = os.path.join(base, root_dir)

        pattern = os.path.join(root, "**", "recording.npz")
        self.recording_files: List[str] = sorted(glob.glob(pattern, recursive=True))
        if not self.recording_files:
            raise FileNotFoundError(f"No recording.npz found under {root}")

        self.num_points = num_points
        self.samples: List[Tuple[str, int]] = []  # (file_path, grasp_idx)

        # Caches ── one copy per Python process
        self.pc_cache: Dict[str, np.ndarray] = {}    # file_path -> (P, 3)
        self.data_cache: Dict[str, np.lib.npyio.NpzFile] = {}  # file_path -> npz

        # Build index table + pre-sample each mesh once
        for f in self.recording_files:
            data = np.load(f)  # only once here
            num_grasps = int(data["grasps"].shape[0])

            mesh_path = os.path.join(os.path.dirname(f), "mesh.obj")
            self.pc_cache[f] = sample_mesh_as_pc(mesh_path, num_points)

            for gi in range(num_grasps):
                self.samples.append((f, gi))

    # --------------------------------------------------------------------- magic
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        file_path, grasp_idx = self.samples[idx]

        # Memory-cache the npz
        if file_path not in self.data_cache:
            self.data_cache[file_path] = np.load(file_path)

        data = self.data_cache[file_path]
        grasp_vec = data["grasps"][grasp_idx].astype(np.float32)  # (19,)
        score = float(data["scores"][grasp_idx])                  # scalar
        pc = self.pc_cache[file_path]                             # (P, 3)

        return (
            torch.from_numpy(pc),             # (P, 3)
            torch.from_numpy(grasp_vec),      # (19,)
            torch.tensor(score, dtype=torch.float32),
        )


# ────────────────────────────────────────────────────────────────────────────────
# DataModule
# ────────────────────────────────────────────────────────────────────────────────
class DLRHand2DataModule(pl.LightningDataModule):
    """
    *All* loaders currently share the same dataset object; only the shuffling
    flag changes.  Pass `num_workers > 0` with care – every worker gets its own
    copy of the two RAM caches.
    """

    def __init__(
        self,
        root_dir: str = "data/grasp_sample/03948459",
        batch_size: int = 4,
        num_workers: int = 0,
        num_points: int = 2048,
        ssd_cache_dir: str = "/mnt/disks/ssd/dataset",
        use_ssd_cache: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset: DLRHand2Dataset | None = None

    # ------------------------------------------------------------------- setup
    def setup(self, stage: str | None = None) -> None:  # noqa: D401
        """Instantiate the underlying Dataset exactly once per rank."""
        if self.dataset is None:
            if self.hparams.use_ssd_cache:
                dataset_path = self.prepare_ssd_cache()
            else:
                dataset_path = self.hparams.root_dir

            self.dataset = DLRHand2Dataset(
                root_dir=dataset_path,
                num_points=self.hparams.num_points,
            )
    
    def prepare_ssd_cache(self) -> str:
        src_root = os.path.join(get_original_cwd(), self.hparams.root_dir)
        dst_root = self.hparams.ssd_cache_dir

        if not os.path.exists(dst_root):
            os.makedirs(dst_root, exist_ok=True)

        # Only copy .npz and mesh files — optionally, you can limit to a subset
        for src_file in glob.glob(os.path.join(src_root, "**", "*.*"), recursive=True):
            # if not src_file.endswith((".npz")):
            if not src_file.endswith((".npz", ".obj")):
                continue
            rel_path = os.path.relpath(src_file, src_root)
            dst_file = os.path.join(dst_root, rel_path)

            if not os.path.exists(dst_file):
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                shutil.copy2(src_file, dst_file)

        return dst_root


    # ---------------------------------------------------------------- loaders
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
            shuffle=False,                     # deterministic!
            num_workers=self.hparams.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        # Uses the same deterministic loader as validation for now
        return self.val_dataloader()
