#!/usr/bin/env python3
"""
extract_pointclouds.py  —  local‐dir & GCS‐zip sampling, no pickle errors
Defaults:
  * densities = [2048,10240,25000]
  * jobs = max(1, cpu_count()//2)
Usage (local unpacked):
  scripts/extract_pointclouds.py \
    --src_dir /mnt/disks/ssd/shapenet/unpacked/02747177/02747177 \
    --dst_bucket adlr2025-pointclouds
Usage (direct zip from GCS):
  scripts/extract_pointclouds.py \
    --src_zip gs://shapenet_bucket/02747177.zip \
    --dst_bucket adlr2025-pointclouds
"""
import argparse
import os
import subprocess
import tempfile
import glob
from pathlib import Path
import numpy as np
import open3d as o3d
from multiprocessing import Pool, cpu_count, get_context
from tqdm import tqdm

# Silence all but errors in Open3D
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
# Prevent any BLAS/MKL oversubscription
os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
)

def run(cmd: str):
    print(f"[CMD] {cmd}")
    subprocess.check_call(cmd, shell=True)

def sample_mesh(mesh_file: str, densities: list, out_root: str, src_root: str):
    """
    Read one mesh, sample it at each density, write to out_root preserving tree under src_root.
    Returns list of output paths.
    """
    mesh_id = Path(mesh_file).stem
    rel = os.path.relpath(os.path.dirname(mesh_file), src_root)
    dst_dir = Path(out_root) / rel / mesh_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()

    outs = []
    for N in densities:
        pts = np.asarray(
            mesh.sample_points_uniformly(number_of_points=N).points,
            dtype=np.float32
        )
        f = dst_dir / f"pc_{N}.npz"
        np.savez_compressed(f, points=pts)
        outs.append(str(f))
    return outs

def _wrapper(args):
    # args is a tuple (mesh_file, densities, out_root, src_root)
    return sample_mesh(*args)

def process_dir(src_dir: str, densities: list, dst_bucket: str, jobs: int):
    """
    Sample all meshes under a local directory tree and rsync to GCS.
    """
    cat = Path(src_dir).name
    mesh_files = list(Path(src_dir).rglob("*.obj"))
    if not mesh_files:
        raise RuntimeError(f"No .obj files found under {src_dir}")
    print(f"[{cat}] → {len(mesh_files)} meshes; using {jobs} workers")

    # prepare scratch out directory
    out_root = Path(src_dir).parent / f"{cat}_pc_out"
    if out_root.exists():
        # remove any stale outputs
        import shutil
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    # build arg tuples
    jobs_list = [
        (str(mf), densities, str(out_root), src_dir)
        for mf in mesh_files
    ]

    # run with a spawn-based Pool to avoid pickling issues
    with get_context("spawn").Pool(processes=jobs) as pool:
        for _ in tqdm(pool.imap_unordered(_wrapper, jobs_list),
                      total=len(jobs_list),
                      desc=f"[{cat}] meshes"):
            pass

    # rsync entire out_root to GCS bucket path
    dest = f"gs://{dst_bucket}/{cat}"
    run(f"gsutil -m rsync -r {out_root} {dest}")
    print(f"[{cat}] ✓ done")

def process_zip(zip_uri: str, densities: list, dst_bucket: str, jobs: int):
    """
    Download a GCS ZIP, sample all meshes, and rsync to GCS.
    """
    cat = Path(zip_uri).stem
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        # download
        zipf = tmp / "archive.zip"
        run(f"gsutil -q cp {zip_uri} {zipf}")
        # unzip
        unz = tmp / "unzipped"
        run(f"unzip -q {zipf} -d {unz}")

        mesh_files = list(unz.rglob("*.obj"))
        if not mesh_files:
            raise RuntimeError(f"No .obj files in {zip_uri}")
        print(f"[{cat}] → {len(mesh_files)} meshes; using {jobs} workers")

        out_root = tmp / "out"
        out_root.mkdir()

        jobs_list = [
            (str(mf), densities, str(out_root), str(unz))
            for mf in mesh_files
        ]

        with get_context("spawn").Pool(processes=jobs) as pool:
            for _ in tqdm(pool.imap_unordered(_wrapper, jobs_list),
                          total=len(jobs_list),
                          desc=f"[{cat}] meshes"):
                pass

        dest = f"gs://{dst_bucket}/{cat}"
        run(f"gsutil -m rsync -r {out_root} {dest}")
        print(f"[{cat}] ✓ done")

def main():
    ap = argparse.ArgumentParser(description="Sample ShapeNet pointclouds")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--src_dir", help="Local unpacked OBJ directory")
    g.add_argument("--src_zip", help="GCS URI of ZIP (e.g. gs://bucket/01234567.zip)")
    ap.add_argument("--dst_bucket", required=True, help="GCS bucket name")
    ap.add_argument(
        "--densities",
        default="2048,10240,25000",
        help="Comma-separated list of point counts"
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=max(1, cpu_count()//2),
        help="Number of parallel workers"
    )

    args = ap.parse_args()
    dens = [int(x) for x in args.densities.split(",")]

    if args.src_dir:
        process_dir(args.src_dir, dens, args.dst_bucket, args.jobs)
    else:
        process_zip(args.src_zip, dens, args.dst_bucket, args.jobs)

if __name__ == "__main__":
    main()
