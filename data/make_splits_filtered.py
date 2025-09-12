#!/usr/bin/env python3
# python -m data.make_splits  $DLR_HAND_ROOT  configs/splits  --pack A --seed 0 --budgets 1 10 25 100
"""
Create 1 %, 10 %, 25 % and 100 % manifests *at the object level* and
derive extra splits for grasp-level generalisation.

This fork additionally **filters out any grasp whose quality score is below
``SCORE_THRESHOLD`` (default: ``1.5``) everywhere** before the train/val/test
splits are produced.  The dataset .npz files are expected to contain either a
``scores`` or a ``qualities`` array with shape *(n_grasps,)*.  If neither key
is present an explicit error is raised so that missing fields cannot go
unnoticed.

Folder layout (four levels):
  ``root / <synsetID> / <objectID> / <sceneID> / recording.npz``
"""

from __future__ import annotations
import argparse, json, os, glob, random, math, pathlib
from collections import defaultdict
from typing import List
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ────────────────────────────────────────────────────────────────────────────────
HELD_OUT_SYNSETS = {"03948459", "02954340"}
SUBSET_PERCENTS  = [0.01, 0.10, 0.25, 1.00]      # 1 %, 10 %, 25 %, 100 %
VAL_PCT_OBJ      = 0.10                          # objects
TEST_PCT_OBJ     = 0.10                          # objects
TEST_PCT_SCENE   = 0.10                          # scenes per *seen* object
SEED             = 42                             # can be overridden by CLI
PATH_PREFIX      = "studentGrasping/student_grasps_v1"
SCORE_THRESHOLD  = 1.5                           #  ←  filter low-quality grasps

# ────────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ────────────────────────────────────────────────────────────────────────────────

def list_scenes(root: str):
    """Return two dictionaries mapping objects / synsets to **filtered** samples.

    A *sample* is a tuple ``(relative_path, grasp_index)`` where
    ``relative_path`` includes the project-level ``PATH_PREFIX`` so that it can
    be written directly into the JSON manifests.
    """
    rec_paths = glob.glob(os.path.join(root, "**", "recording.npz"), recursive=True)
    by_obj, by_syn = defaultdict(list), defaultdict(list)

    for rp in rec_paths:
        tail = os.path.relpath(rp, root)            # "02808440/…/recording.npz"
        syn, obj, scene = tail.split(os.sep)[:3]    # correct IDs
        rel_str = os.path.join(PATH_PREFIX, tail)   # prefixed path for JSON

        # ── load grasp quality scores and keep only the high ones
        with np.load(rp) as d:
            if "scores" in d:
                scores = d["scores"]
            elif "qualities" in d:                  # fall-back alias
                scores = d["qualities"]
            else:
                raise KeyError(
                    f"{rp} does not contain a 'scores' or 'qualities' array—"
                    "cannot perform quality filtering."
                )

            high_idx = np.nonzero(scores >= SCORE_THRESHOLD)[0]

        if high_idx.size == 0:
            # no grasp in this scene meets the quality bar – skip entirely
            continue

        samples = [(rel_str, int(gi)) for gi in high_idx]

        key_obj = f"{syn}/{obj}"
        by_obj[key_obj].extend(samples)
        by_syn[syn].extend(samples)

    return by_obj, by_syn


def split_objects(obj_keys, pct_val, pct_test, rng):
    rng.shuffle(obj_keys)
    n_val  = math.ceil(len(obj_keys) * pct_val)
    n_test = math.ceil(len(obj_keys) * pct_test)
    val_objs, test_objs = obj_keys[:n_val], obj_keys[n_val:n_val + n_test]
    train_objs          = obj_keys[n_val + n_test:]
    return train_objs, val_objs, test_objs


def build_manifest(root, rng):
    """Legacy entry point – kept for compatibility with older tooling."""
    by_obj, by_syn = list_scenes(root)
    regular_objs = [k for k in by_obj if k.split("/")[0] not in HELD_OUT_SYNSETS]

    # object-level split
    train_objs, val_objs, test_obj_objs = split_objects(
        regular_objs, VAL_PCT_OBJ, TEST_PCT_OBJ, rng
    )

    # collect samples
    manifest = {k: [] for k in ("train", "val", "test_object", "test_grasp", "test_category")}

    for o in train_objs:
        samples = by_obj[o]
        rng.shuffle(samples)
        n_scene_holdout = math.ceil(len(samples) * TEST_PCT_SCENE)
        manifest["test_grasp"].extend(samples[:n_scene_holdout])
        manifest["train"     ].extend(samples[n_scene_holdout:])

    for o in val_objs:
        manifest["val"].extend(by_obj[o])

    for o in test_obj_objs:
        manifest["test_object"].extend(by_obj[o])

    # held-out categories
    for syn in HELD_OUT_SYNSETS:
        manifest["test_category"].extend(by_syn[syn])

    return manifest


# ────────────────────────────────────────────────────────────────────────────────
#  HELPER – works on pre-filtered look-ups
# ────────────────────────────────────────────────────────────────────────────────

def build_manifest_from_lookup(by_obj, by_syn, rng):
    """Identical to :pyfunc:`build_manifest`, but takes pre-filtered look-ups."""
    regular_objs = [k for k in by_obj if k.split("/")[0] not in HELD_OUT_SYNSETS]
    train_objs, val_objs, test_obj_objs = split_objects(
        regular_objs, VAL_PCT_OBJ, TEST_PCT_OBJ, rng
    )

    manifest = {k: [] for k in ("train", "val", "test_object", "test_grasp", "test_category")}

    for o in train_objs:
        samples = by_obj[o]
        rng.shuffle(samples)
        n_scene_holdout = math.ceil(len(samples) * TEST_PCT_SCENE)
        manifest["test_grasp"].extend(samples[:n_scene_holdout])
        manifest["train"     ].extend(samples[n_scene_holdout:])

    for o in val_objs:
        manifest["val"].extend(by_obj[o])

    for o in test_obj_objs:
        manifest["test_object"].extend(by_obj[o])

    # held-out categories
    for syn in HELD_OUT_SYNSETS:
        manifest["test_category"].extend(by_syn[syn])

    return manifest


# ────────────────────────────────────────────────────────────────────────────────
#  NEW ENTRY POINT WITH CLI BUDGETS + SEED/PACK
# ────────────────────────────────────────────────────────────────────────────────

def label_for_pct(pct: float) -> str:
    """0.01→'1', 0.10→'10', 0.25→'25', 1.0→'100'."""
    return str(int(round(pct * 100)))


def parse_budgets(values: list[str] | None) -> List[float]:
    """Accept '1 10 25 100' or '0.01 0.10 0.25 1.0'; default to SUBSET_PERCENTS."""
    if not values:
        return SUBSET_PERCENTS
    out: List[float] = []
    for v in values:
        try:
            out.append(int(v) / 100.0)  # integers like 1, 10, 25, 100
            continue
        except ValueError:
            pass
        f = float(v)
        if not (0.0 < f <= 1.0):
            raise ValueError(f"Invalid budget '{v}'. Use e.g. 1 10 25 100 or 0.01 0.10 0.25 1.0")
        out.append(f)
    # de-dupe preserving order
    seen = set(); uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x); seen.add(x)
    return uniq


def build_all(root: str, outdir: str, budgets: List[float], seed: int):
    """Build all requested budgets using the given RNG seed."""
    rng = random.Random(seed)

    # ── list once, reuse (already quality-filtered!)
    by_obj, by_syn = list_scenes(root)
    regular_objs = [o for o in by_obj if o.split("/")[0] not in HELD_OUT_SYNSETS]

    total_objs = len(regular_objs)
    print(f"[info] {total_objs} trainable objects found (after quality filter)")

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for pct in budgets:
        # ---------------------------------------------------------- choose objects
        rng.seed(seed)                           # reproducible for every pct
        need   = math.ceil(total_objs * pct)
        chosen = rng.sample(regular_objs, k=need) if pct < 1.0 else list(regular_objs)
        print(f"[info] {pct:4.0%}  →  {len(chosen)} objects")

        # ---------------------------------------------------------- prune helpers
        pruned_by_obj = {k: v for k, v in by_obj.items() if k in chosen}
        pruned_by_syn = defaultdict(list, {syn: by_syn[syn] for syn in HELD_OUT_SYNSETS})
        for k, v in pruned_by_obj.items():
            pruned_by_syn[k.split("/")[0]].extend(v)

        # ---------------------------------------------------------- build manifest
        manifest = build_manifest_from_lookup(pruned_by_obj, pruned_by_syn, rng)

        tag = label_for_pct(pct)
        fn  = outdir / f"split_{tag}.json"
        with open(fn, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[ok]   wrote {fn}")


# ────────────────────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build quality-filtered train/val/test JSON manifests.")
    ap.add_argument("root",   help="Path to DLR–Hand II root (e.g. $DLR_HAND_ROOT)")
    ap.add_argument("outdir", help="Output directory for split JSONs (e.g. configs/splits)")
    ap.add_argument("--pack", choices=["A","B"], default="A",
                    help="Convenience flag: A→seed 0, B→seed 1 (overridden by --seed).")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed to use. If unset: 0 for A, 1 for B.")
    ap.add_argument("--budgets", nargs="*", default=None,
                    help="Budgets as percents ('1 10 25 100') or fractions ('0.01 0.10 0.25 1.0').")
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else (0 if args.pack == "A" else 1)
    budgets = parse_budgets(args.budgets)

    build_all(args.root, args.outdir, budgets, seed)
