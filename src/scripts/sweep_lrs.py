#!/usr/bin/env python3
"""
Sweep exactly SIX runs with `full.yaml` as the base config.

Combos:
    lr_head      : 1e-3 | 2e-3
    lr_backbone  : 1e-5 | 1e-4 | 7e-5
    encoder_unfreeze_step : 0

Each run name:  lrh=<lr_head>_lrb=<lr_backbone>_uf=0
"""

import itertools, subprocess, sys, shlex
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "trainer_joint.py"

LR_HEADS = ["1e-3", "2e-3"]
LR_BBS   = ["1e-5", "1e-4", "7e-5"]
UNFREEZE = 0   # single value → 6 total runs

def run(cmd):
    print("➤", " ".join(shlex.quote(str(c)) for c in cmd))
    subprocess.run(cmd, check=True)

# … unchanged header …

for lr_h, lr_bb in itertools.product(LR_HEADS, LR_BBS):
    name = f"lrh={lr_h}_lrb={lr_bb}_uf={UNFREEZE}"

    cmd = [
        sys.executable, str(SCRIPT),
        "-m",
        "--config-name", "full",
        f"model.lr_head={lr_h}",
        f"model.lr_backbone={lr_bb}",
        f"model.encoder_unfreeze_step={UNFREEZE}",
        # quote the value because it contains '='
        f'logger.name="lrh={lr_h}_lrb={lr_bb}_uf={UNFREEZE}"',
    ]
    run(cmd)
