#!/usr/bin/env python3
"""
Quick quality-gate for split manifests.

Example:
  python src/utils/check_split_scores.py \
         configs/splits/split_25.json \
         data/student_grasping/studentGrasping/student_grasps_v1 \
         --n 200
"""
import json, random, argparse
from pathlib import Path
import numpy as np

PATH_PREFIX     = "studentGrasping/student_grasps_v1"   # must match manifest
SCORE_THRESHOLD = 1.5

def load_scores(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as d:
        if "scores" in d:
            return d["scores"]
        elif "qualities" in d:
            return d["qualities"]
        else:
            raise KeyError(f"{npz_path} lacks 'scores' or 'qualities' array")

def main(manifest_json, dataset_root, n):
    with open(manifest_json) as f:
        mani = json.load(f)

    # aggregate all buckets
    pool = sum((
        mani["train"],
        mani["val"],
        mani["test_object"],
        mani["test_grasp"],
        mani["test_category"],
    ), [])

    random.seed(0)
    sample = random.sample(pool, k=min(n, len(pool)))

    offenders = []
    for rel_path, gi in sample:
        npz = Path(dataset_root) / Path(rel_path).relative_to(PATH_PREFIX)
        score = load_scores(npz)[gi]
        if score < SCORE_THRESHOLD:
            offenders.append((rel_path, gi, float(score)))

    print(f"Checked {len(sample)} samples from {Path(manifest_json).name}")
    if offenders:
        print(f"❌  {len(offenders)} below {SCORE_THRESHOLD}:")
        for rel, gi, sc in offenders:
            print(f"   {rel:80s}  idx {gi:>3}  score={sc:.3f}")
    else:
        print(f"✅  All scores ≥ {SCORE_THRESHOLD}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_json")
    ap.add_argument("dataset_root")
    ap.add_argument("--n", type=int, default=200,
                    help="Number of random grasps to check (default 200)")
    args = ap.parse_args()
    main(args.manifest_json, args.dataset_root, args.n)
