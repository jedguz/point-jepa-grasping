#!/usr/bin/env python3
"""
DLR-Hand-2 joint-angle regression DataModule

• preload_all = False  →  lazy, fast start-up (debug runs)
• preload_all = True   →  eager load only the recordings referenced
                         in the manifest (full training)

Splits are defined by a JSON manifest produced by data/make_splits.py
"""

from __future__ import annotations
import os, glob, shutil, json, math
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import numpy as np
import torch, pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from hydra.utils import get_original_cwd, to_absolute_path   # ← HERE
import trimesh

from datasets.manifest_dataset import ManifestDataset

def _sample_mesh_as_pc(mesh_path: str, n: int = 2048) -> np.ndarray:
    """
    Load an OBJ, collapse scenes → single Trimesh, then fps-sample `n` points.
    Works for watertight or non-watertight meshes.
    """
    mesh = trimesh.load(mesh_path, process=False)

    # If the loader returns a Scene (multi-part mesh), merge all geometries
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"[mesh-load] no geometry in '{mesh_path}'")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    # Fallback: convex hull for open meshes (sampling needs a surface)
    if not getattr(mesh, "is_watertight", False):
        mesh = mesh.convex_hull

    pts = mesh.sample(n)                     # (n,3) numpy.float64
    return pts.astype(np.float32)
# ───────────────────────────────── heavyweight dataset
class DLRHand2JointDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 num_points: int = 2048,
                 ssd_cache_dir: str = "/mnt/disks/ssd/cache",
                 *,
                 preload_all: bool = False,
                 preload_only: Set[str] | None = None):
        super().__init__()

        # resolve dataset root (Hydra-friendly & absolute)
        if os.path.isabs(root_dir):
            self.source_root = root_dir
        else:
            try:
                self.source_root = os.path.join(get_original_cwd(), root_dir)
            except ValueError:
                self.source_root = os.path.abspath(root_dir)
        self.source_root = os.path.abspath(self.source_root)

        self.ssd_cache_root = os.path.abspath(ssd_cache_dir)
        os.makedirs(self.ssd_cache_root, exist_ok=True)
        self.num_points = num_points

        # gather recording paths
        if preload_only:
            recs = sorted(list(preload_only))
        else:
            recs = sorted(
                glob.glob(os.path.join(self.source_root, "**", "recording.npz"),
                          recursive=True)
            )
        if not recs:
            raise FileNotFoundError(f"No recordings under {self.source_root}")
        print(f"[INFO] Found {len(recs)} recording files")

        # list all (rec_path, grasp_idx)
        self.samples: List[Tuple[str, int]] = []
        for rp in recs:
            with np.load(rp) as d:
                n = d["grasps"].shape[0]
            self.samples.extend((rp, gi) for gi in range(n))

        # caches
        self.pc_cache:   Dict[str, np.ndarray] = {}
        self.data_cache: Dict[str, Dict[str, np.ndarray]] = {}

        # decide what to preload
        eager = set()
        if preload_all:
            eager = preload_only if preload_only else set(recs)

        if eager:
            print(f"[INFO] Preloading {len(eager)} recordings …")
            self._preload(eager)
        else:
            print("[INFO] Using lazy loading (no upfront preload)")

    # preload helper --------------------------------------------------------
    def _preload(self, rec_paths: Set[str]):
        for rp in rec_paths:
            c_npz  = self._cache(rp)
            c_mesh = self._cache(os.path.join(os.path.dirname(rp), "mesh.obj"))

            if c_npz not in self.pc_cache:
                self.pc_cache[c_npz] = _sample_mesh_as_pc(c_mesh, self.num_points)

            if c_npz not in self.data_cache:
                with np.load(c_npz) as d:
                    self.data_cache[c_npz] = {
                        "grasps": d["grasps"].copy(),
                        "scores": d["scores"].copy(),
                    }

    # copy-to-cache helper ---------------------------------------------------
    def _cache(self, src: str) -> str:
        rel = os.path.relpath(src, self.source_root)
        dst = os.path.join(self.ssd_cache_root, rel)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        return dst

    # dunder ----------------------------------------------------------------
    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        rp, gi = self.samples[idx]
        c_npz  = self._cache(rp)

        if c_npz not in self.pc_cache:
            c_mesh = self._cache(os.path.join(os.path.dirname(rp), "mesh.obj"))
            self.pc_cache[c_npz] = _sample_mesh_as_pc(c_mesh, self.num_points)

        if c_npz not in self.data_cache:
            with np.load(c_npz) as d:
                self.data_cache[c_npz] = {
                    "grasps": d["grasps"].copy(),
                    "scores": d["scores"].copy(),
                }

        data, pc = self.data_cache[c_npz], self.pc_cache[c_npz]
        g = data["grasps"][gi].astype(np.float32)
        pose, joints = g[:7], g[7:]
        score = float(data["scores"][gi])

        return {"points": torch.from_numpy(pc),
                "pose":   torch.from_numpy(pose),
                "joints": torch.from_numpy(joints),
                "meta":   {"synset": os.path.basename(os.path.dirname(rp)),
                           "score": score}}

# ───────────────────────────────── Lightning DataModule
class DLRHand2JointDataModule(pl.LightningDataModule):
    def __init__(self, *,
                 root_dir: str,
                 split_file: str,
                 ssd_cache_dir: str = "/mnt/disks/ssd/cache",
                 batch_size: int = 8,
                 num_workers: int = 0,
                 num_points: int = 2048,
                 score_temp: float = 0.0,
                 preload_all: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self._base:   DLRHand2JointDataset | None = None
        self._splits: Dict[str, Dataset] = {}

    # ---------------------------------------------------------------- setup
    def setup(self, stage=None):
        if self._base is not None:
            return

        # manifest (always resolve relative to project root)
        split_file = to_absolute_path(self.hparams.split_file)
        with open(split_file) as f:
            manifest = json.load(f)

        # dataset root (same rule)
        root_abs = to_absolute_path(self.hparams.root_dir)
        root_tail = os.path.join(*root_abs.split(os.sep)[-4:])  # last 4 dirs

        # ───────── helper: turn manifest path → absolute on disk ──────────
        def canonical(p: str) -> str:
            return to_absolute_path(p)          # ← one and done

        needed_recs = {canonical(p) for split in manifest.values()
                                   for p, _ in split}

        # build dataset
        self._base = DLRHand2JointDataset(
            root_dir      = root_abs,            # USE ABSOLUTE PATH
            num_points    = self.hparams.num_points,
            ssd_cache_dir = self.hparams.ssd_cache_dir,
            preload_all   = self.hparams.preload_all,
            preload_only  = needed_recs,
        )

        # slice
        self._splits = {k: ManifestDataset(v, self._base) for k, v in manifest.items()}
        for k, ds in self._splits.items():
            print(f"[INFO] split '{k}': {len(ds)} samples")

        print(f"[DEBUG] preload_all from Hydra → {self.hparams.preload_all}")


    # ---------------------------------------------------------------- loader
    def _loader(self, name: str, shuffle: bool):
        ds = self._splits[name]
        sampler = None
        if self.hparams.score_temp > 0:
            w = [math.exp(s["meta"]["score"] / self.hparams.score_temp) for s in ds]
            sampler, shuffle = WeightedRandomSampler(w, len(ds), True), False
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def train_dataloader         (self): return self._loader("train", True)
    def val_dataloader           (self): return self._loader("val",   False)
    def test_dataloader          (self): return self._loader("test_object", False)
    def test_grasp_dataloader    (self): return self._loader("test_grasp",  False)
    def test_category_dataloader (self): return self._loader("test_category", False)
