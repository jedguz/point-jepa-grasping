#!/usr/bin/env python3
#python -m data.make_splits        data/student_grasping/studentGrasping/student_grasps_v1        configs/splits
"""
Create 25 %, 50 % and 100 % manifests *at the object level* and
derive extra splits for grasp-level generalisation.

Folder layout (four levels):
  root / <synsetID> / <objectID> / <sceneID> / recording.npz
"""

from __future__ import annotations
import argparse, json, os, glob, random, math, pathlib
from collections import defaultdict
import numpy as np

## DONT CHANGE HOLDOUT SETS
HELD_OUT_SYNSETS = {"03948459", "02954340"}
SUBSET_PERCENTS  = [0.1, 0.25, 0.50, 1.00]      # 10%, 25 %, 50 %, 100 %
VAL_PCT_OBJ      = 0.10                    # objects
TEST_PCT_OBJ     = 0.10                    # objects
TEST_PCT_SCENE   = 0.10                    # scenes per *seen* object
SEED             = 42
PATH_PREFIX = "studentGrasping/student_grasps_v1"

def list_scenes(root: str):
    rec_paths = glob.glob(os.path.join(root, "**", "recording.npz"),
                          recursive=True)
    by_obj, by_syn = defaultdict(list), defaultdict(list)

    for rp in rec_paths:
        tail = os.path.relpath(rp, root)            # "02808440/…/recording.npz"
        syn, obj, scene = tail.split(os.sep)[:3]    # correct IDs

        rel_str = os.path.join(PATH_PREFIX, tail)   # prefixed path for JSON

        key_obj = f"{syn}/{obj}"
        with np.load(rp) as d:
            n_grasps = d["grasps"].shape[0]
        samples = [(rel_str, gi) for gi in range(n_grasps)]

        by_obj[key_obj].extend(samples)
        by_syn[syn].extend(samples)

    return by_obj, by_syn

def split_objects(obj_keys, pct_val, pct_test, rng):
    rng.shuffle(obj_keys)
    n_val  = math.ceil(len(obj_keys)*pct_val)
    n_test = math.ceil(len(obj_keys)*pct_test)
    val_objs, test_objs = obj_keys[:n_val], obj_keys[n_val:n_val+n_test]
    train_objs          = obj_keys[n_val+n_test:]
    return train_objs, val_objs, test_objs

def build_manifest(root, rng):
    by_obj, by_syn = list_scenes(root)
    regular_objs = [k for k in by_obj if k.split("/")[0] not in HELD_OUT_SYNSETS]

    # object-level split
    train_objs, val_objs, test_obj_objs = split_objects(regular_objs,
                                                        VAL_PCT_OBJ, TEST_PCT_OBJ, rng)

    # collect samples
    manifest = {k: [] for k in
                ("train", "val", "test_object", "test_grasp", "test_category")}

    for o in train_objs:
        samples = by_obj[o]
        rng.shuffle(samples)
        n_scene_holdout = math.ceil(len(samples)*TEST_PCT_SCENE)
        manifest["test_grasp"].extend(samples[:n_scene_holdout])
        manifest["train"].extend   (samples[n_scene_holdout:])

    for o in val_objs:
        manifest["val"].extend(by_obj[o])

    for o in test_obj_objs:
        manifest["test_object"].extend(by_obj[o])

    # held-out categories
    for syn in HELD_OUT_SYNSETS:
        manifest["test_category"].extend(by_syn[syn])

    return manifest

def build_all(root, outdir):
    rng = random.Random(SEED)

    # ── list once, reuse
    by_obj, by_syn = list_scenes(root)
    regular_objs = [o for o in by_obj if o.split("/")[0] not in HELD_OUT_SYNSETS]

    total_objs = len(regular_objs)
    print(f"📦  {total_objs} trainable objects found")

    for pct in SUBSET_PERCENTS:
        # ---------------------------------------------------------- choose objects
        rng.seed(SEED)         # reproducible for every pct
        need = math.ceil(total_objs * pct)
        chosen = rng.sample(regular_objs, k=need) if pct < 1.0 else regular_objs
        print(f"  • {pct:4.0%}  →  {len(chosen)} objects")

        # ---------------------------------------------------------- prune helpers
        pruned_by_obj = {k: v for k, v in by_obj.items() if k in chosen}
        pruned_by_syn = defaultdict(list, {syn: by_syn[syn] for syn in HELD_OUT_SYNSETS})
        for k, v in pruned_by_obj.items():
            pruned_by_syn[k.split("/")[0]].extend(v)
        # ---------------------------------------------------------- build manifest
        manifest = build_manifest_from_lookup(
            pruned_by_obj, pruned_by_syn, rng
        )

        tag = {0.10: "10", 0.25: "25", 0.50: "50", 1.00: "100"}[pct]
        fn  = pathlib.Path(outdir) / f"split_{tag}.json"
        with open(fn, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"     ✅  wrote {fn}")


# ------------------------------------------------------------ helper = old build_manifest
def build_manifest_from_lookup(by_obj, by_syn, rng):
    """Identical to old build_manifest(), but takes pre-filtered look-ups."""
    regular_objs = [k for k in by_obj if k.split("/")[0] not in HELD_OUT_SYNSETS]
    train_objs, val_objs, test_obj_objs = split_objects(
        regular_objs, VAL_PCT_OBJ, TEST_PCT_OBJ, rng
    )

    manifest = {k: [] for k in
                ("train", "val", "test_object", "test_grasp", "test_category")}

    for o in train_objs:
        samples = by_obj[o]
        rng.shuffle(samples)
        n_scene_holdout = math.ceil(len(samples) * TEST_PCT_SCENE)
        manifest["test_grasp"].extend(samples[:n_scene_holdout])
        manifest["train"].extend(samples[n_scene_holdout:])

    for o in val_objs:
        manifest["val"].extend(by_obj[o])

    for o in test_obj_objs:
        manifest["test_object"].extend(by_obj[o])

    # held-out categories
    for syn in HELD_OUT_SYNSETS:
        manifest["test_category"].extend(by_syn[syn])

    return manifest

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root",   help="data/student_grasping/studentGrasping/student_grasps_v1")
    p.add_argument("outdir", help="configs/splits")
    args = p.parse_args()
    build_all(args.root, args.outdir)
