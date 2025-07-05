"""
Lightning DataModule for **joint-angle regression** on the DLR-Hand-2 dataset.
Keeps the original SSD caching logic but returns (points, 7-D pose, 12-D joints).
Supports optional sorting and filtering of samples by score per model.
"""

from __future__ import annotations
import os, glob, shutil
from typing import List, Tuple
from collections import defaultdict
import numpy as np
import torch, pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from hydra.utils import get_original_cwd
import trimesh


# ───────────────────────────────────────────────────────────────────────────────
# Helper
# ───────────────────────────────────────────────────────────────────────────────
def _sample_mesh_as_pc(mesh_path: str, num_points: int = 2048) -> np.ndarray:
    mesh = trimesh.load(mesh_path, process=False)
    if not mesh.is_watertight:
        mesh = mesh.convex_hull
    pts, _ = mesh.sample(num_points, return_index=True)
    return pts.astype(np.float32)


# ───────────────────────────────────────────────────────────────────────────────
# Dataset
# ───────────────────────────────────────────────────────────────────────────────
class DLRHand2JointDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        num_points: int = 2048,
        ssd_cache_dir: str = "/mnt/disks/ssd/cache",
        top_percentile: float = 0.7,
        sort_by_score: bool = False,
    ) -> None:
        super().__init__()

        if os.path.isabs(root_dir):
            ds_root = root_dir
        else:
            ds_root = os.path.join(get_original_cwd(), root_dir)

        self.source_root = os.path.abspath(ds_root)
        self.ssd_cache_root = os.path.abspath(ssd_cache_dir)
        os.makedirs(self.ssd_cache_root, exist_ok=True)
        self.num_points = num_points
        self.top_percentile = top_percentile
        self.sort_by_score = sort_by_score

        # collect all recordings
        pattern = os.path.join(self.source_root, "**", "recording.npz")
        self.recording_files = sorted(glob.glob(pattern, recursive=True))
        print(f"[INFO] Found {len(self.recording_files)} recording files")
        if not self.recording_files:
            raise FileNotFoundError(f"No recordings found under {self.source_root}")

        self.samples: List[Tuple[str, int]] = []
        self.pc_cache, self.data_cache = {}, {}

        if self.sort_by_score:
            # Group recordings by model ID (parent folder of the 0-9 folder)
            recordings_by_model = defaultdict(list)
            for rec_path in self.recording_files:
                # model_id = two levels up from recording.npz
                model_id = os.path.basename(os.path.dirname(os.path.dirname(rec_path)))
                recordings_by_model[model_id].append(rec_path)

            # For each model, collect and sort grasps by score descending
            for model_id, rec_paths in recordings_by_model.items():
                grasp_entries = []
                for rec_path in rec_paths:
                    data = np.load(rec_path)
                    scores = data["scores"]
                    for gi, score in enumerate(scores):
                        grasp_entries.append((score, rec_path, gi))
                grasp_entries.sort(key=lambda x: -x[0])
                num_keep = int(len(grasp_entries) * self.top_percentile)
                for _, rec_path, gi in grasp_entries[:num_keep]:
                    self.samples.append((rec_path, gi))
        else:
            for f in self.recording_files:
                num_grasps = int(np.load(f)["grasps"].shape[0])
                for gi in range(num_grasps):
                    self.samples.append((f, gi))

    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # -------------------------------------------------------------------------
    def __getitem__(self, idx: int):
        rec_path, gi = self.samples[idx]

        # SSD mirror caching
        cached_npz = self._cache(rec_path)
        mesh_path  = self._cache(os.path.join(os.path.dirname(rec_path), "mesh.obj"))

        # lazy load caches
        if cached_npz not in self.data_cache:
            self.data_cache[cached_npz] = np.load(cached_npz)
        data = self.data_cache[cached_npz]

        if cached_npz not in self.pc_cache:
            self.pc_cache[cached_npz] = _sample_mesh_as_pc(mesh_path, self.num_points)

        pc     = self.pc_cache[cached_npz]                                # (N,3)
        g_full = data["grasps"][gi].astype(np.float32)                    # (19,)
        pose, joints = g_full[:7], g_full[7:]                             # (7,) (12,)
        score  = float(data["scores"][gi])

        return {
            "points": torch.from_numpy(pc),           # (N,3)
            "pose":   torch.from_numpy(pose),         # (7,)
            "joints": torch.from_numpy(joints),       # (12,)
            "meta":   {
                "synset": os.path.basename(os.path.dirname(rec_path)),  # e.g. model ID
                "score": score,
            },
        }

    # -------------------------------------------------------------------------
    def _cache(self, src: str) -> str:
        rel = os.path.relpath(src, self.source_root)
        dst = os.path.join(self.ssd_cache_root, rel)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        return dst


# ───────────────────────────────────────────────────────────────────────────────
# DataModule wrapper
# ───────────────────────────────────────────────────────────────────────────────
class DLRHand2JointDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        ssd_cache_dir: str = "/mnt/disks/ssd/cache",
        batch_size: int = 8,
        num_workers: int = 2,
        num_points: int = 2048,
        top_percentile: float = 0.7,
        sort_by_score: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset: DLRHand2JointDataset | None = None

    # -------------------------------------------------------------------------
    def setup(self, stage: str | None = None):
        print("\n ---SET UP DATA MODULE---: \n")
        if self.dataset is None:
            ds_root = os.path.join(get_original_cwd(), self.hparams.root_dir)
            self.dataset = DLRHand2JointDataset(
                root_dir      = ds_root,
                num_points    = self.hparams.num_points,
                ssd_cache_dir = self.hparams.ssd_cache_dir,
                top_percentile   = self.hparams.top_percentile,
                sort_by_score = self.hparams.sort_by_score,
            )
        print(f"[INFO] Final dataset contains {len(self.dataset)} samples")

    # -------------------------------------------------------------------------
    def _loader(self, shuffle: bool):
        return DataLoader(
            self.dataset,
            batch_size  = self.hparams.batch_size,
            shuffle     = shuffle,
            num_workers = self.hparams.num_workers,
        )

    def train_dataloader(self): return self._loader(shuffle=True)
    def val_dataloader  (self): return self._loader(shuffle=False)
    def test_dataloader (self): return self.val_dataloader()
