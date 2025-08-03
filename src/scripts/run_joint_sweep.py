#!/usr/bin/env python3
import itertools, subprocess, sys, shlex
from pathlib import Path

# ► 1. point to the correct file
SCRIPT = str(Path(__file__).resolve().parent / "trainer_joint.py")

SPLIT_IDS = ["01", "02", "05", "10", "25", "50", "100"]
SEEDS     = [0, 1]
BACKBONES = [True, False]

def run(cmd):
    print("➤", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)

for split_id, use_bb, seed in itertools.product(SPLIT_IDS, BACKBONES, SEEDS):
    tag  = "jepa" if use_bb else "scratch"
    name = f"{tag}_split{split_id}_seed{seed}"

    split_path = Path(f"configs/splits/split_{split_id}.json")
    if not split_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {split_path}")

    cmd = [
        sys.executable, SCRIPT, "-m",
        f"data.split_file={split_path}",
        "+trainer=steps",                        # ► 2. append config group
        "data.preload_all=false",           #  ← HERE
        f"ckpt.load_backbone={str(use_bb).lower()}",
        f"train.seed={seed}",
        f"logger.name={name}",
    ]
    run(cmd)
