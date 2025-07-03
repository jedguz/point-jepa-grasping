#!/usr/bin/env python3
"""download_grasp_categories.py

Download **studentGrasping_v1.tar.gz** from Google Cloud Storage (using **gsutil** so it honours
your current *gcloud* login) and copy a *subset* of categories/objects into your
local `grasp_sample` folder with the expected ShapeNet‑style structure:

```
<DEST_ROOT>/<category_id>/<object_id>/<grasp_instance_folders>
```

Why another update?
-------------------
The archive’s internal layout turned out to be:

```
studentGrasping/          # 1st component
└── student_grasps_v1/    # 2nd component
    └── 03211117/         # ← category  (e.g. drills)
        └── <object>/
            └── <grasps>/
```

(Previously we assumed the top‑level folder was *just* `studentGrasping_v1`,
so the script couldn’t find your categories.)

Key fixes
---------
* Detects either `student_grasps_v1` **or** `studentGrasping_v1` inside the tar.
* Strips the correct **two** leading folders when extracting, so you get
  `grasp_sample/<cat>/<obj>/…` no matter the archive prefix.
* Uses `args.dest_root` directly—no more mysterious `parent.parent` hacks.

Usage example
-------------
```bash
python download_grasp_categories.py \
  --dest-root /home/jguzeel_gmail_com/jed-repo/ADLR/data/grasp_sample \
  --categories 03211117 02818832 03761084 \
  --num-objects 7
```

If a category isn’t in the tar, you’ll still get a clear warning.

"""
from __future__ import annotations

import argparse
import pathlib
import random
import shutil
import subprocess
import sys
import tarfile
from typing import Dict, List

HTTP_URL = (
    "https://storage.googleapis.com/adlr2025-pointclouds/grasps/"
    "student_grasps_v1/studentGrasping_v1.tar.gz"
)
GS_URL = HTTP_URL.replace("https://storage.googleapis.com/", "gs://")
DEFAULT_TAR = pathlib.Path("~/studentGrasping_v1.tar.gz").expanduser()

# -----------------------------------------------------------------------------
# gsutil helpers
# -----------------------------------------------------------------------------

def download_tar_with_gsutil(gs_uri: str, dest: pathlib.Path) -> None:
    if dest.exists():
        print(f"[INFO] Using cached archive → {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["gsutil", "cp", "-n", gs_uri, str(dest)]
    print("[INFO] Downloading with gsutil …")
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        sys.exit("[ERROR] gsutil not found. Install Cloud SDK or add it to PATH.")
    except subprocess.CalledProcessError as exc:
        sys.exit(f"[ERROR] gsutil failed ({exc.returncode}).")
    print("[INFO] Download complete.")

# -----------------------------------------------------------------------------
# Tar inspection / extraction helpers
# -----------------------------------------------------------------------------

def find_prefix_idx(parts: tuple[str, ...]) -> int | None:
    """Return index *after* the archive prefix (i.e. where *cat* starts)."""
    for marker in ("student_grasps_v1", "studentGrasping_v1"):
        if marker in parts:
            return parts.index(marker) + 1
    return None


def catalogue_members(tf: tarfile.TarFile, categories: List[str]) -> Dict[str, Dict[str, List[tarfile.TarInfo]]]:
    cat_map: Dict[str, Dict[str, List[tarfile.TarInfo]]] = {}
    for m in tf.getmembers():
        parts = pathlib.Path(m.name).parts
        idx = find_prefix_idx(parts)
        if idx is None or len(parts) <= idx + 1:
            continue  # Not a category/object path
        cat, obj = parts[idx], parts[idx + 1]
        if cat not in categories:
            continue
        cat_map.setdefault(cat, {}).setdefault(obj, []).append(m)
    return cat_map


def safe_extract(tf: tarfile.TarFile, members: List[tarfile.TarInfo], dest_root: pathlib.Path, strip_parts: int) -> None:
    for m in members:
        rel_parts = pathlib.Path(m.name).parts[strip_parts:]
        out_path = dest_root.joinpath(*rel_parts)
        if m.isdir():
            out_path.mkdir(parents=True, exist_ok=True)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tf.extractfile(m) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a subset of the studentGrasping_v1 dataset")
    parser.add_argument("--dest-root", required=True, type=pathlib.Path, help="Destination root (ShapeNet style)")
    parser.add_argument("--categories", nargs="+", required=True, help="Category IDs to include")
    parser.add_argument("--num-objects", type=int, default=7, help="Objects to sample per category (default: 7)")
    parser.add_argument("--tar", type=pathlib.Path, default=DEFAULT_TAR, help="Where to cache the downloaded tarball")
    args = parser.parse_args()

    download_tar_with_gsutil(GS_URL, args.tar)

    with tarfile.open(args.tar) as tf:
        cat_map = catalogue_members(tf, args.categories)
        for cat in args.categories:
            objs = list(cat_map.get(cat, {}).keys())
            if not objs:
                print(f"[WARN] Category {cat} not present in archive!")
                continue
            pick = random.sample(objs, min(args.num_objects, len(objs)))
            print(f"[INFO] {cat}: extracting {len(pick)} of {len(objs)} objects …")
            # prefix length is constant for this archive (2 components):
            prefix_len = 2  # studentGrasping/*
            for obj in pick:
                safe_extract(tf, cat_map[cat][obj], args.dest_root, strip_parts=prefix_len)
                print(f"  ↳ {args.dest_root / cat / obj}")

    print("[DONE] Subset ready →", args.dest_root)


if __name__ == "__main__":
    main()
