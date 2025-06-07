#!/usr/bin/env python3
"""
Point-cloud sampler, 7 Jun 2025 revision
* Mirrors the original ZIP folder tree so files never overwrite.
* Streams uploads with `gsutil rsync -m` (parallel).
* Uses local SSD if present (/mnt/disks/ssd0) for fast I/O.
* Halves Open3D’s internal thread use to avoid CPU oversub.
"""
import argparse, os, subprocess, tempfile, glob, shutil, multiprocessing as mp
from pathlib import Path
import numpy as np, open3d as o3d
from tqdm import tqdm

# keep Open3D chatty = OFF
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
os.environ.update(
    OMP_NUM_THREADS="1",      # OpenBLAS / numpy
    OPENBLAS_NUM_THREADS="1", # "
    MKL_NUM_THREADS="1"       # if MKL sneaks in
)

def run(cmd):
    print(cmd); subprocess.check_call(cmd, shell=True)

def sample_mesh(job):
    mesh_file, densities, out_root, unzip_root = job
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    rel_parent = os.path.relpath(os.path.dirname(mesh_file), unzip_root)
    mesh_id = Path(mesh_file).stem
    dst_dir = Path(out_root, rel_parent, mesh_id)
    dst_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    for N in densities:
        pts = np.asarray(
            mesh.sample_points_uniformly(number_of_points=N).points,
            dtype=np.float32,
        )
        out_file = dst_dir / f"pc_{N}.npz"
        np.savez_compressed(out_file, points=pts)
        outs.append(out_file)
    return outs

def process_zip(zip_uri, densities, dst_bucket, workers):
    cat = Path(zip_uri).stem
    scratch = Path("/mnt/disks/ssd0/tmp") if Path("/mnt/disks/ssd0").exists() \
              else Path(tempfile.gettempdir())
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        tmp = Path(tmp)
        archive = tmp / "archive.zip"
        run(f"gsutil -q cp {zip_uri} {archive}")
        run(f"unzip -q {archive} -d {tmp/'unzipped'}")

        mesh_files = glob.glob(f"{tmp/'unzipped'}/**/*.obj", recursive=True)
        if not mesh_files:
            raise RuntimeError(f"no .obj in {zip_uri}")
        print(f"[{cat}] sampling {len(mesh_files)} meshes with {workers} workers")

        jobs = [(mf, densities, tmp/'out', tmp/'unzipped') for mf in mesh_files]
        with mp.get_context("spawn").Pool(processes=workers) as pool:
            list(tqdm(pool.imap_unordered(sample_mesh, jobs), total=len(jobs)))

        dest = f"gs://{dst_bucket}/{cat}"
        run(f"gsutil -m rsync -r {tmp/'out'} {dest}")
        print(f"[{cat}] ✓ done")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_zip", help="gs://bucket/01234567.zip", required=True)
    ap.add_argument("--dst_bucket", help="destination bucket name", required=True)
    ap.add_argument("--densities", default="2048,10240,25000",
                    help="comma-sep counts")
    ap.add_argument("--jobs", type=int,
                    default=max(1, os.cpu_count() // 2))
    args = ap.parse_args()
    dens = [int(x) for x in args.densities.split(",")]
    process_zip(args.src_zip, dens, args.dst_bucket, args.jobs)
