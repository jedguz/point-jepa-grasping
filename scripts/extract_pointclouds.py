# File: scripts/extract_pointclouds.py
"""
Script: extract_pointclouds.py
Location: place this file in the project root under the `scripts/` directory.

Description:
Downloads a ShapeNet category ZIP from your GCS bucket, unpacks all
`.obj` meshes (preserving subdirectory structure), samples point clouds at
configured densities for each mesh *in parallel*, and uploads the resulting
`.npz` files to your existing GCS bucket:

  gs://<DST_BUCKET>/<category_id>/<mesh_id>/pc_<N>.npz

Prerequisite:
 1. Ensure the destination bucket `adlr2025-pointclouds` exists:
      gcloud storage buckets create gs://adlr2025-pointclouds \
        --project=adlr2025 --location=europe-west3 --storage-class=STANDARD
 2. VM scopes: use `--scopes=https://www.googleapis.com/auth/cloud-platform`.

Usage:
python scripts/extract_pointclouds.py \
    --project adlr2025 \
    --src_bucket shapenet_bucket \
    --category_id 02747177 \
    --dst_bucket adlr2025-pointclouds \
    --densities 2048 10240 51200 \
    --jobs 16

Alternate (direct ZIP URI):
python scripts/extract_pointclouds.py \
    --project adlr2025 \
    --src_zip gs://shapenet_bucket/02747177.zip \
    --dst_bucket adlr2025-pointclouds \
    --densities 2048 10240 51200 \
    --jobs 16

Options:
  --jobs N     Number of parallel worker processes (default: CPU count)

This enables full CPU utilization for mesh sampling.
"""
import argparse
import os
import subprocess
import tempfile
import glob
import open3d as o3d
import numpy as np
from multiprocessing import Pool, cpu_count


def run(cmd):
    """Run a shell command, raising on error."""
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True)


def sample_and_upload_mesh(args):
    mesh_file, category, densities, dst_bucket = args
    # Extract object ID: assume structure .../<category>/<object_id>/models/...obj
    # Mesh file lives in .../<object_id>/models/*.obj, so parent parent dir is object_id
    object_dir = os.path.dirname(mesh_file)
    mesh_parent = os.path.dirname(object_dir)
    mesh_id = os.path.basename(mesh_parent)
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    with tempfile.TemporaryDirectory() as tmpm:
        for N in densities:
            pts = np.asarray(mesh.sample_points_uniformly(number_of_points=N).points)
            # Save under a per-mesh tmp folder to avoid naming collisions
            local_dir = os.path.join(tmpm, mesh_id)
            os.makedirs(local_dir, exist_ok=True)
            npz_tmp = os.path.join(local_dir, f"pc_{N}.npz")
            np.savez_compressed(npz_tmp, points=pts)
            gcs_path = f"gs://{dst_bucket}/{category}/{mesh_id}/pc_{N}.npz"
            run(f"gsutil -q cp {npz_tmp} {gcs_path}")
            print(f"Uploaded {gcs_path}")f"Uploaded {gcs_path}")


def process_zip(zip_source, category, densities, dst_bucket, jobs):
    # 1) Download ZIP
    with tempfile.TemporaryDirectory() as tmp:
        zip_name = os.path.basename(zip_source)
        zip_path = os.path.join(tmp, zip_name)
        run(f"gsutil -q cp {zip_source} {zip_path}")

        # 2) Unzip all .obj meshes preserving folder structure
        mesh_dir = os.path.join(tmp, category)
        os.makedirs(mesh_dir, exist_ok=True)
        run(f"unzip -q {zip_path} '*.obj' -d {mesh_dir}")

        # 3) Gather all mesh files
        mesh_files = glob.glob(os.path.join(mesh_dir, '**', '*.obj'), recursive=True)
        if not mesh_files:
            raise RuntimeError(f"No .obj files found in {zip_source}")

        # 4) Parallel sampling and upload
        work_items = [(mf, category, densities, dst_bucket) for mf in mesh_files]
        with Pool(processes=jobs) as pool:
            pool.map(sample_and_upload_mesh, work_items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--densities",
    type=lambda s: [int(item) for item in s.split(",")],
    default=[2048, 10240, 51200],
    help="Comma-separated point counts to sample, e.g. 2048,10240,51200"
)
    parser.add_argument(
        "--jobs", type=int, default=cpu_count(),
        help="Number of parallel worker processes (default: CPU count)"
    )
    args = parser.parse_args()

    os.environ['GOOGLE_CLOUD_PROJECT'] = args.project

    # Build ZIP source list and determine category
    if args.src_zip:
        zip_list = [args.src_zip]
        category = args.category_id or os.path.splitext(os.path.basename(args.src_zip))[0]
    else:
        if not args.category_id:
            parser.error("--category_id is required when using --src_bucket")
        zip_list = [f"gs://{args.src_bucket}/{args.category_id}.zip"]
        category = args.category_id

    # Process each ZIP
    for zs in zip_list:
        process_zip(
            zip_source=zs,
            category=category,
            densities=args.densities,
            dst_bucket=args.dst_bucket,
            jobs=args.jobs
        )

if __name__ == '__main__':
    main()
