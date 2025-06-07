#!/usr/bin/env python3
"""
extract_pointclouds.py

Modes:
  --src_zip  : Download & sample directly from a GCS ZIP
  --src_dir  : Sample from a local unpacked OBJ directory

Features:
  - Default densities: 2048,10240,25000
  - Silence Open3D INFO logs
  - Environment vars to limit BLAS threads
  - Live progress bars via tqdm
  - Default workers: half of available CPUs
  - Uses multiprocessing with spawn to avoid pickling issues
  - Streams upload via gsutil rsync
"""
import argparse
import os
import subprocess
import tempfile
import glob
from pathlib import Path
import numpy as np
import open3d as o3d
from multiprocessing import get_context, cpu_count
from tqdm import tqdm

# Silence Open3D logs to errors only
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
# Limit BLAS/MKL threads to 1 each
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

def run(cmd: str):
    """Run shell command, raising on error"""
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)


def sample_mesh(mesh_file: str, densities: list, out_root: str, src_root: str):
    """
    Sample a single mesh at given densities.
    Writes pc_<N>.npz under out_root/<relative_path>/<mesh_id>/
    Returns list of output file paths.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    # Determine relative parent directory to mirror tree
    rel = Path(mesh_file).parent.relative_to(src_root)
    mesh_id = Path(mesh_file).stem
    dst_dir = Path(out_root) / rel / mesh_id
    dst_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for N in densities:
        pts = np.asarray(
            mesh.sample_points_uniformly(number_of_points=N).points,
            dtype=np.float32
        )
        out_file = dst_dir / f"pc_{N}.npz"
        np.savez_compressed(out_file, points=pts)
        outputs.append(str(out_file))
    return outputs


def process_dir(src_dir: str, densities: list, dst_bucket: str, workers: int):
    """Process a local directory of OBJ meshes."""
    src_root = Path(src_dir)
    cat = src_root.name
    mesh_files = [str(p) for p in src_root.rglob("*.obj")]
    if not mesh_files:
        raise RuntimeError(f"No .obj files in {src_dir}")
    print(f"[{cat}] Sampling {len(mesh_files)} meshes with {workers} workers...")
    out_root = src_root / "out"
    jobs = [(mf, densities, str(out_root), str(src_root)) for mf in mesh_files]
    ctx = get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        for _ in tqdm(pool.imap_unordered(
                lambda args: sample_mesh(*args), jobs),
                total=len(jobs), desc=f"[{cat}] meshes"):
            pass
    # Upload via rsync -m
    dest = f"gs://{dst_bucket}/{cat}"
    run(f"gsutil -m rsync -r {out_root} {dest}")
    print(f"[{cat}] ✓ done")


def process_zip(zip_uri: str, densities: list, dst_bucket: str, workers: int):
    """Download, unzip, sample, and upload for a GCS zip URI."""
    cat = Path(zip_uri).stem
    print(f"[{cat}] Processing ZIP {zip_uri} ...")
    # Use local SSD if available for scratch
    scratch = Path("/mnt/disks/ssd0/tmp") if Path("/mnt/disks/ssd0").exists() else Path(tempfile.gettempdir())
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(scratch)) as tmpdir:
        tmp = Path(tmpdir)
        archive = tmp / "archive.zip"
        unzipped = tmp / "unzipped"
        out_root = tmp / "out"
        # Download and unzip
        run(f"gsutil -q cp {zip_uri} {archive}")
        run(f"unzip -q {archive} -d {unzipped}")
        mesh_files = [str(p) for p in unzipped.rglob("*.obj")]
        if not mesh_files:
            raise RuntimeError(f"No .obj files found in {zip_uri}")
        print(f"[{cat}] Sampling {len(mesh_files)} meshes with {workers} workers...")
        jobs = [(mf, densities, str(out_root), str(unzipped)) for mf in mesh_files]
        ctx = get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for _ in tqdm(pool.imap_unordered(
                    lambda args: sample_mesh(*args), jobs),
                    total=len(jobs), desc=f"[{cat}] meshes"):
                pass
        # Upload via rsync
        dest = f"gs://{dst_bucket}/{cat}"
        run(f"gsutil -m rsync -r {out_root} {dest}")
        print(f"[{cat}] ✓ done")


def main():
    parser = argparse.ArgumentParser(description="Extract & sample pointclouds from ShapeNet meshes.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--src_zip", help="GCS URI of the category zip (gs://bucket/01234567.zip)")
    group.add_argument("--src_dir", help="Local directory of unpacked OBJ meshes")
    parser.add_argument("--dst_bucket", required=True, help="Destination GCS bucket name")
    parser.add_argument("--densities", default="2048,10240,25000",
                        help="Comma-separated point counts (default: 2048,10240,25000)")
    parser.add_argument("--jobs", type=int, default=max(1, cpu_count()//2),
                        help="Number of parallel mesh workers (default=half CPUs)")
    args = parser.parse_args()
    dens = [int(x) for x in args.densities.split(",")]
    if args.src_dir:
        process_dir(args.src_dir, dens, args.dst_bucket, args.jobs)
    else:
        process_zip(args.src_zip, dens, args.dst_bucket, args.jobs)

if __name__ == '__main__':
    main()
