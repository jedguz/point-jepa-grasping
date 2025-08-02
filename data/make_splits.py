#!/usr/bin/env python3
# python -m data.make_splits \
#   data/student_grasping/studentGrasping/student_grasps_v1 \
#   configs/splits
"""
Low-data manifests with a *fixed* evaluation suite.

Splits produced per subset:
  • train         –   object-level (subset-dependent)
  • val           –   10 % objects, fixed across all subsets
  • test_object   –   10 % objects, fixed
  • test_category –   all objects in HELD_OUT_SYNSETS, fixed

Grasps with quality/score < SCORE_THRESHOLD are discarded everywhere.
Folder layout:
  root / <synsetID> / <objectID> / <sceneID> / recording.npz
"""

from __future__ import annotations
import argparse, json, os, glob, random, math, pathlib
from collections import defaultdict
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────────────────────────────────────
HELD_OUT_SYNSETS = {"03948459", "02954340"}               # unseen categories
SUBSET_PERCENTS  = [0.01, 0.02, 0.05, 0.10, 0.15,
                    0.25, 0.50, 1.00]                      # 1 % … 100 %
VAL_PCT_OBJ      = 0.15                                    # 15 % objects
TEST_PCT_OBJ     = 0.15                                    # 15 % objects
SEED             = 42
PATH_PREFIX      = "studentGrasping/student_grasps_v1"
SCORE_THRESHOLD  = 1.5

# ────────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────────────────────────────────────
def list_scenes(root: str):
    """Return look-ups (by object / synset) after score filtering."""
    recs = glob.glob(os.path.join(root, "**", "recording.npz"), recursive=True)
    by_obj, by_syn = defaultdict(list), defaultdict(list)

    for rp in recs:
        tail = os.path.relpath(rp, root)                    # "028_…/recording.npz"
        syn, obj, _ = tail.split(os.sep)[:3]
        rel_json = os.path.join(PATH_PREFIX, tail)          # stored in manifest

        with np.load(rp) as d:
            if "scores" in d:        scores = d["scores"]
            elif "qualities" in d:   scores = d["qualities"]
            else:
                raise KeyError(f"{rp} lacks 'scores' / 'qualities'")

        keep = np.nonzero(scores >= SCORE_THRESHOLD)[0]
        if keep.size == 0:
            continue  # no good grasps

        samples = [(rel_json, int(i)) for i in keep]
        obj_key = f"{syn}/{obj}"
        by_obj[obj_key].extend(samples)
        by_syn[syn].extend(samples)

    return by_obj, by_syn


def split_objects(obj_keys, pct_val, pct_test, rng):
    """Random object-level split; percentages refer to *obj_keys* length."""
    rng.shuffle(obj_keys)
    n_val  = math.ceil(len(obj_keys) * pct_val)
    n_test = math.ceil(len(obj_keys) * pct_test)
    val_objs   = obj_keys[: n_val]
    test_objs  = obj_keys[n_val : n_val + n_test]
    train_objs = obj_keys[n_val + n_test :]
    return train_objs, val_objs, test_objs


def stratified_sample(objects_by_syn, need, rng):
    """Pick ≈need objects, ≥1 per synset (proportional otherwise)."""
    total = sum(len(v) for v in objects_by_syn.values())
    chosen = []

    for syn, objs in objects_by_syn.items():
        share = max(1, int(round(len(objs) / total * need)))
        chosen.extend(rng.sample(objs, k=min(share, len(objs))))

    if len(chosen) < need:
        remaining = [o for lst in objects_by_syn.values() for o in lst
                     if o not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: need - len(chosen)])

    return chosen[:need]   # trim overflow


# ────────────────────────────────────────────────────────────────────────────────
#  MANIFEST BUILD
# ────────────────────────────────────────────────────────────────────────────────
def build_manifests(root: str, outdir: str):
    rng = random.Random(SEED)

    # ① list scenes once (quality-filtered)
    by_obj, by_syn = list_scenes(root)
    regular_objs   = [o for o in by_obj if o.split("/")[0] not in HELD_OUT_SYNSETS]
    objects_by_syn = defaultdict(list)
    for o in regular_objs:
        objects_by_syn[o.split("/")[0]].append(o)

    total_objs = len(regular_objs)
    print(f"📦  {total_objs} trainable objects after filtering")

    # ② build ONE global evaluation split (val + test_object)
    train_pool, val_objs, test_obj_objs = split_objects(
        regular_objs, VAL_PCT_OBJ, TEST_PCT_OBJ, rng
    )

    eval_manifest = {k: [] for k in ("val", "test_object", "test_category")}
    for o in val_objs:
        eval_manifest["val"].extend(by_obj[o])
    for o in test_obj_objs:
        eval_manifest["test_object"].extend(by_obj[o])
    for syn in HELD_OUT_SYNSETS:
        eval_manifest["test_category"].extend(by_syn[syn])

    # ③ build LOW-DATA subsets (only *train* changes)
    for pct in SUBSET_PERCENTS:
        rng.seed(SEED)                                   # reproducible per-pct
        if pct < 1.0:
            need   = math.ceil(len(train_pool) * pct)
            chosen = stratified_sample(objects_by_syn, need, rng)
        else:
            chosen = train_pool

        print(f"  • {pct:5.2%} → {len(chosen)} train objects")

        manifest = {k: [] for k in ("train", *eval_manifest.keys())}
        for k in eval_manifest:                          # fixed eval sets
            manifest[k].extend(eval_manifest[k])

        for o in chosen:                                 # subset-specific train
            manifest["train"].extend(by_obj[o])

        tag = f"{int(round(pct * 100)):02d}"             # 0.01 → "01", 1.0 → "100"
        fn  = pathlib.Path(outdir) / f"split_{tag}.json"
        with open(fn, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"     ✅  wrote {fn}")


# ────────────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build fixed-evaluation JSON manifests for multiple data regimes"
    )
    ap.add_argument("root",
                    help="data/student_grasping/studentGrasping/student_grasps_v1")
    ap.add_argument("outdir", help="configs/splits")
    args = ap.parse_args()

    build_manifests(args.root, args.outdir)
