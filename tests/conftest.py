# tests/conftest.py
import sys
from pathlib import Path

# ── make `scripts.*` importable for every test run ───────────────────────
ROOT = Path(__file__).resolve().parents[1]   # → .../ADLR
if str(ROOT) not in sys.path:                # avoid duplicates
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch
from pathlib import Path

from scripts.dlrhand2_datamodule import DLRHand2Dataset, DLRHand2DataModule
from scripts.grasp_regressor import GraspRegressor


@pytest.fixture(scope="session")
def tmp_dataset(tmp_path_factory):
    """
    Build a tiny synthetic dataset in a temporary directory:

    * one mesh.obj (unit cube via trimesh)
    * one recording.npz with 5 random grasps/scores
    """
    import trimesh

    root = tmp_path_factory.mktemp("dlrhand2_fake")
    obj_dir = root / "foo"
    obj_dir.mkdir(parents=True, exist_ok=True)

    # minimal mesh
    trimesh.primitives.Box().export(obj_dir / "mesh.obj")

    # fake recordings
    rng = np.random.default_rng(0)
    grasps = rng.random((5, 19)).astype(np.float32)
    scores = rng.random(5).astype(np.float32)

    np.savez(obj_dir / "recording.npz", grasps=grasps, scores=scores)
    return root
