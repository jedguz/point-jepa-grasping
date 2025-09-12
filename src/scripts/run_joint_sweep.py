#!/usr/bin/env python3
"""
Sweep launcher for joint training with Pack-aware splits.

Defaults:
  - Pack A: seed=0, budgets: 1,10,25,100  (configs/splitsA/split_*.json)
  - Pack B: seed=1, budgets: 25,100       (configs/splitsB/split_*.json)

Examples:
  # Run pack A only (default)
  ./sweep_joint.py

  # Run pack B only (anchors 25,100)
  ./sweep_joint.py --packs B

  # Run both packs
  ./sweep_joint.py --packs A B

  # Override budgets for pack A
  ./sweep_joint.py --packs A --budgets 1 10

  # Override seeds explicitly
  ./sweep_joint.py --seedA 123 --seedB 456
"""
import argparse
import itertools
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "trainer_joint.py"

# Defaults per pack
PACK_DEFAULTS = {
    "A": {"seed": 0, "budgets": ["1", "10", "25", "100"], "dir": "splitsA"},
    "B": {"seed": 1, "budgets": ["25", "100"],             "dir": "splitsB"},
}

BACKBONES = [True, False]  # True = JEPA fine-tune; False = scratch

def run(cmd):
    print("➤", " ".join(shlex.quote(str(c)) for c in cmd))
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(description="Pack-aware sweep for joint training.")
    ap.add_argument("--packs", nargs="+", default=["A"], choices=["A", "B"],
                    help="Which packs to run (default: A).")
    ap.add_argument("--budgets", nargs="*", default=None,
                    help="Override budgets for ALL packs (e.g., 1 10 25 100).")
    ap.add_argument("--seedA", type=int, default=None, help="Override seed for pack A (default: 0).")
    ap.add_argument("--seedB", type=int, default=None, help="Override seed for pack B (default: 1).")
    ap.add_argument("--splits_root", default="configs",
                    help="Root folder containing splitsA/ and splitsB/ (default: configs).")
    ap.add_argument("--config-name", default="full",
                    help="Hydra config name (without .yaml). Default: full")
    ap.add_argument("--max_steps", default="25000", help="trainer.max_steps override.")
    ap.add_argument("--val_check_interval", default="5000", help="trainer.val_check_interval override.")
    # Optional LR overrides (kept here for reproducibility)
    ap.add_argument("--lr_head", default="1.5e-3")
    ap.add_argument("--lr_backbone", default="1.5e-5")
    args = ap.parse_args()

    # Resolve pack settings
    pack_to_seed = {
        "A": (args.seedA if args.seedA is not None else PACK_DEFAULTS["A"]["seed"]),
        "B": (args.seedB if args.seedB is not None else PACK_DEFAULTS["B"]["seed"]),
    }
    pack_to_budgets = {
        p: (args.budgets if args.budgets else PACK_DEFAULTS[p]["budgets"]) for p in ["A", "B"]
    }
    splits_root = Path(args.splits_root)

    for pack in args.packs:
        seed = pack_to_seed[pack]
        budgets = pack_to_budgets[pack]
        split_dir = (splits_root / PACK_DEFAULTS[pack]["dir"]).resolve()

        for split, use_bb in itertools.product(budgets, BACKBONES):
            tag  = "jepa" if use_bb else "scratch"
            name = f"NEWSPLIT_{pack}_{tag}_split{split}_seed{seed}"

            split_path = (split_dir / f"split_{split}.json").resolve()
            if not split_path.is_file():
                raise FileNotFoundError(f"Missing split file: {split_path}")

            cmd = [
                sys.executable, str(SCRIPT),
                "--config-name", args.config-name if hasattr(args, "config-name") else args.config_name,  # safety
                f"data.split_file={split_path}",
                f"model.lr_head={args.lr_head}",
                f"model.lr_backbone={args.lr_backbone}",
                "model.encoder_unfreeze_step=0",              # FT immediately (scratch ignores)
                f"ckpt.load_backbone={str(use_bb).lower()}",
                f"trainer.max_steps={args.max_steps}",
                f"trainer.val_check_interval={args.val_check_interval}",
                f"train.seed={seed}",
                f"logger.name={name}",
            ]
            run(cmd)

if __name__ == "__main__":
    main()
