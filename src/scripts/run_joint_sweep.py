#!/usr/bin/env python3
"""
16-run sweep:
  splits      : 01 | 05 | 25 | 100
  seeds       : 0 | 1
  modes       : JEPA fine-tune (load_backbone=true) vs scratch
"""
import itertools, subprocess, sys, shlex
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "trainer_joint.py"

SPLITS = ["01", "05", "25", "100"]
SEEDS  = [0, 1]
BACKBONES = [True, False]   # True = JEPA FT, False = scratch

def run(cmd):
    print("➤", " ".join(shlex.quote(str(c)) for c in cmd))
    subprocess.run(cmd, check=True)

for split, use_bb, seed in itertools.product(SPLITS, BACKBONES, SEEDS):
    tag  = "jepa" if use_bb else "scratch"
    name = f"{tag}_split{split}_seed{seed}"
    split_path = Path(f"configs/splits/split_{split}.json")
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    cmd = [
        sys.executable, str(SCRIPT), "-m",
        # data
        f"data.split_file={split_path}",
        # model
        "model.lr_head=1e-3",
        "model.lr_backbone=1e-5",
        "model.encoder_unfreeze_step=0",     # fine-tune immediately (scratch ignores)
        f"ckpt.load_backbone={str(use_bb).lower()}",
        # trainer
        "trainer.max_steps=12000",
        "trainer.val_check_interval=2000",
        # misc
        f"train.seed={seed}",
        f"logger.name={name}",
    ]
    run(cmd)
