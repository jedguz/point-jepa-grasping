# src/scripts/dlrhand2_joint_datamodule.py
"""
Lightning DataModule for **joint-angle regression** on the DLR-Hand-2 dataset.
Keeps the original SSD caching logic but returns (points, 7-D pose, 12-D joints).
"""

from __future__ import annotations
import os, glob, shutil
from typing import List, Tuple
import numpy as np
import torch, pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from hydra.utils import get_original_cwd
import trimesh
from collections import defaultdict


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
        top_percentile: float = 1.0,
        sort_by_score: bool = True,
    ) -> None:
        super().__init__()
        self.source_root = os.path.abspath(root_dir)
        self.ssd_cache_root = os.path.abspath(ssd_cache_dir)
        os.makedirs(self.ssd_cache_root, exist_ok=True)
        self.num_points = num_points
        self.top_percentile = top_percentile
        self.sort_by_score = sort_by_score

        # collect all recordings
        pattern = os.path.join(self.source_root, "**", "recording.npz")
        self.recording_files = sorted(glob.glob(pattern, recursive=True))
        if not self.recording_files:
            raise FileNotFoundError(f"No recordings found under {self.source_root}")

        # Collect samples and group by model
        model_samples = defaultdict(list)
        
        for f in self.recording_files:
            # Get number of grasps and model info
            with np.load(f) as data:
                num_grasps = int(data["grasps"].shape[0])
                scores = data["scores"]
                
                # Extract model info from path
                path_parts = f.split(os.sep)
                synset = model_id = None
                for i, part in enumerate(path_parts):
                    if part.isdigit() and len(part) == 8:
                        synset = part
                        if i + 1 < len(path_parts):
                            model_id = path_parts[i + 1]
                        break
                
                if synset and model_id:
                    model_key = f"{synset}_{model_id}"
                    for gi in range(num_grasps):
                        score = float(scores[gi])
                        model_samples[model_key].append((f, gi, score))

        # Sort and filter samples per model
        self.samples = []
        for model_key, samples_list in model_samples.items():
            if self.sort_by_score:
                samples_list.sort(key=lambda x: x[2], reverse=True)
            
            if self.top_percentile < 1.0:
                n_keep = max(1, int(len(samples_list) * self.top_percentile))
                samples_list = samples_list[:n_keep]
            
            # Add to final samples (remove score)
            for f, gi, score in samples_list:
                self.samples.append((f, gi))

        self.pc_cache, self.data_cache = {}, {}
        
        print(f"Dataset initialized with {len(self.samples)} samples from {len(model_samples)} models")
        print(f"Settings: sort_by_score={self.sort_by_score}, top_percentile={self.top_percentile}")
        
        # Debug: show sample counts for first few models
        sample_counts = {}
        for model_key, samples_list in list(model_samples.items())[:3]:
            original_count = len(samples_list)
            if self.sort_by_score:
                samples_list.sort(key=lambda x: x[2], reverse=True)
            if self.top_percentile < 1.0:
                n_keep = max(1, int(len(samples_list) * self.top_percentile))
                kept_count = n_keep
            else:
                kept_count = len(samples_list)
            sample_counts[model_key] = (original_count, kept_count)
        
        for model_key, (orig, kept) in sample_counts.items():
            print(f"  {model_key}: {orig} -> {kept} samples")

    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # -------------------------------------------------------------------------
    def __getitem__(self, idx: int):
        rec_path, gi = self.samples[idx]

        # SSD mirror
        cached_npz = self._cache(rec_path)
        mesh_path  = self._cache(os.path.join(os.path.dirname(rec_path), "mesh.obj"))

        # Load data and immediately extract what we need
        if cached_npz not in self.data_cache:
            with np.load(cached_npz) as npz_data:
                self.data_cache[cached_npz] = {
                    'grasps': npz_data['grasps'].copy(),
                    'scores': npz_data['scores'].copy()
                }
        data = self.data_cache[cached_npz]

        if cached_npz not in self.pc_cache:
            self.pc_cache[cached_npz] = _sample_mesh_as_pc(mesh_path, self.num_points)

        pc     = self.pc_cache[cached_npz]                                # (N,3)
        g_full = data["grasps"][gi].astype(np.float32)                    # (19,)
        pose, joints = g_full[:7], g_full[7:]                             # (7,) (12,)
        score  = float(data["scores"][gi])

        # update to record category and score
        return {
            "points": torch.from_numpy(pc),           # (N,3)
            "pose":   torch.from_numpy(pose),         # (7,)
            "joints": torch.from_numpy(joints),       # (12,)
            "meta":   {
                "synset": os.path.basename(os.path.dirname(rec_path)),  # e.g. "03948459"
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
        sort_by_score: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.dataset: DLRHand2JointDataset | None = None

    # -------------------------------------------------------------------------
    def setup(self, stage: str | None = None):
        if self.dataset is None:
            ds_root = os.path.join(get_original_cwd(), self.hparams.root_dir)
            self.dataset = DLRHand2JointDataset(
                root_dir     = ds_root,
                num_points   = self.hparams.num_points,
                ssd_cache_dir= self.hparams.ssd_cache_dir,
                top_percentile= self.hparams.top_percentile,
                sort_by_score = self.hparams.sort_by_score,
            )

    # -------------------------------------------------------------------------
    def _loader(self, shuffle: bool):
        return DataLoader(
            self.dataset,
            batch_size  = self.hparams.batch_size,
            shuffle     = shuffle,
            num_workers = self.hparams.num_workers,
            pin_memory  = torch.cuda.is_available(),
        )

    def train_dataloader(self): return self._loader(shuffle=True)
    def val_dataloader  (self): return self._loader(shuffle=False)
    def test_dataloader (self): return self.val_dataloader()