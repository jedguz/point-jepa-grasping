#!/usr/bin/env python3
"""
Scan all recording.npz files under /mnt/disks/ssd and print
basic statistics + an ASCII histogram of the `scores` vector.
"""

import os, glob, json, time
import numpy as np
from collections import Counter
from tqdm import tqdm

DATA_ROOT = "data/student_grasping"            # adjust if needed
PATTERN   = "**/recording.npz"          # matches your dataset layout
NUM_BINS  = 25                          # histogram resolution
CUTS      = [1.0, 1.5, 2.0, 2.5, 3.0]   # candidate thresholds to report

t0 = time.time()
all_scores = []

rec_paths = glob.glob(os.path.join(DATA_ROOT, PATTERN), recursive=True)
print(f"Found {len(rec_paths):,} recording files. Loading…")

for rp in tqdm(rec_paths):
    try:
        with np.load(rp) as d:
            all_scores.append(d["scores"].astype(np.float32))
    except PermissionError:
        # skip files we can't open
        continue
    except Exception as e:
        # skip any other loading errors, but log them
        print(f"Skipped {rp}: {e}")
        continue

if not all_scores:
    print("No scores loaded—check your DATA_ROOT and file permissions!")
    exit(1)

scores = np.concatenate(all_scores)

# ─── basic stats ─────────────────────────────────────────────────────
print(f"\nLoaded {scores.shape[0]:,} grasps in {time.time()-t0:.1f}s")
print(f"min={scores.min():.2f}  median={np.median(scores):.2f}  "
      f"mean={scores.mean():.2f}  max={scores.max():.2f}")

# percent below each candidate cut-off
for c in CUTS:
    frac = (scores < c).mean() * 100
    print(f"  < {c:4.1f}  → {frac:5.1f}% of grasps")

# ─── ASCII histogram ─────────────────────────────────────────────────
hist, bin_edges = np.histogram(scores, bins=NUM_BINS,
                               range=(scores.min(), scores.max()))
max_bar = 40
print("\nHistogram:")
for cnt, left, right in zip(hist, bin_edges[:-1], bin_edges[1:]):
    bar = "█" * int(max_bar * cnt / hist.max())
    print(f"{left:4.1f} – {right:4.1f} | {bar} {cnt}")
