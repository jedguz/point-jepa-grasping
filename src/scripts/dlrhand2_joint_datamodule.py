"""
Lightning DataModule for **joint-angle regression** on the DLR-Hand-2 dataset.
Preloads point clouds and .npz data to avoid file handle issues.
Returns (points, 7-D pose, 12-D joints).
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


def _sample_mesh_as_pc(mesh_path: str, num_points: int = 2048) -> np.ndarray:
    mesh = trimesh.load(mesh_path, process=False)
    if not mesh.is_watertight:
        mesh = mesh.convex_hull
    pts, _ = mesh.sample(num_points, return_index=True)
    return pts.astype(np.float32)


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

        ds_root = root_dir if os.path.isabs(root_dir) else os.path.join(get_original_cwd(), root_dir)
        self.source_root = os.path.abspath(ds_root)
        self.ssd_cache_root = os.path.abspath(ssd_cache_dir)
        os.makedirs(self.ssd_cache_root, exist_ok=True)

        self.num_points = num_points
        self.top_percentile = top_percentile
        self.sort_by_score = sort_by_score
        self.pc_cache, self.data_cache = {}, {}
        self.samples: List[Tuple[str, int]] = []

        # Gather recordings
        pattern = os.path.join(self.source_root, "**", "recording.npz")
        self.recording_files = sorted(glob.glob(pattern, recursive=True))
        print(f"[INFO] Found {len(self.recording_files)} recording files")
        if not self.recording_files:
            raise FileNotFoundError(f"No recordings found under {self.source_root}")

        # Filter and sort samples
        if self.sort_by_score:
            recordings_by_model = defaultdict(list)
            for rec_path in self.recording_files:
                model_id = os.path.basename(os.path.dirname(os.path.dirname(rec_path)))
                recordings_by_model[model_id].append(rec_path)

            for model_id, rec_paths in recordings_by_model.items():
                grasp_entries = []
                for rec_path in rec_paths:
                    with np.load(rec_path) as data:
                        scores = data["scores"]
                        for gi, score in enumerate(scores):
                            grasp_entries.append((score, rec_path, gi))
                grasp_entries.sort(key=lambda x: -x[0])
                num_keep = int(len(grasp_entries) * self.top_percentile)
                self.samples.extend((rec_path, gi) for _, rec_path, gi in grasp_entries[:num_keep])
        else:
            for f in self.recording_files:
                with np.load(f) as data:
                    num_grasps = data["grasps"].shape[0]
                self.samples.extend((f, gi) for gi in range(num_grasps))

        # Preload point clouds and npz content
        print(f"[INFO] Preloading {len(self.samples)} samples...")
        seen_paths = set()
        for rec_path, _ in self.samples:
            if rec_path in seen_paths:
                continue
            seen_paths.add(rec_path)

            cached_npz = self._cache(rec_path)
            mesh_path = self._cache(os.path.join(os.path.dirname(rec_path), "mesh.obj"))

            # Load and cache point cloud
            if cached_npz not in self.pc_cache:
                self.pc_cache[cached_npz] = _sample_mesh_as_pc(mesh_path, self.num_points)

            # Load and cache npz content safely (copy to close file)
            with np.load(cached_npz) as data:
                self.data_cache[cached_npz] = {
                    "grasps": data["grasps"].copy(),
                    "scores": data["scores"].copy()
                }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec_path, gi = self.samples[idx]
        cached_npz = self._cache(rec_path)
        data = self.data_cache[cached_npz]
        pc = self.pc_cache[cached_npz]

        g_full = data["grasps"][gi].astype(np.float32)
        pose, joints = g_full[:7], g_full[7:]
        score = float(data["scores"][gi])

        return {
            "points": torch.from_numpy(pc),       # (N, 3)
            "pose":   torch.from_numpy(pose),     # (7,)
            "joints": torch.from_numpy(joints),   # (12,)
            "meta": {
                "synset": os.path.basename(os.path.dirname(rec_path)),
                "score": score,
            },
        }

    def _cache(self, src: str) -> str:
        rel = os.path.relpath(src, self.source_root)
        dst = os.path.join(self.ssd_cache_root, rel)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        return dst


class DLRHand2JointDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        ssd_cache_dir: str = "/mnt/disks/ssd/cache",
        batch_size: int = 8,
        num_workers: int = 0,  # SAFER: prevent file handle issues
        num_points: int = 2048,
        top_percentile: float = 0.7,
        sort_by_score: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset: DLRHand2JointDataset | None = None

    def setup(self, stage: str | None = None):
        if self.dataset is None:
            ds_root = os.path.join(get_original_cwd(), self.hparams.root_dir)
            self.dataset = DLRHand2JointDataset(
                root_dir=ds_root,
                num_points=self.hparams.num_points,
                ssd_cache_dir=self.hparams.ssd_cache_dir,
                top_percentile=self.hparams.top_percentile,
                sort_by_score=self.hparams.sort_by_score,
            )
            print(f"[INFO] Final dataset contains {len(self.dataset)} samples")

    def _loader(self, shuffle: bool):
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
        )

    def train_dataloader(self): return self._loader(shuffle=True)
    def val_dataloader(self):   return self._loader(shuffle=False)
    def test_dataloader(self):  return self.val_dataloader()
