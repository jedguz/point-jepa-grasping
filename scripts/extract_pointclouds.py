# File: scripts/extract_pointclouds.py
"""
Script: extract_pointclouds.py
Location: place this file in the project root under the `scripts/` directory.

Description:
Downloads ShapeNet category ZIPs from your GCS bucket, unpacks all
`.obj` meshes (preserving subdirectory structure), samples point clouds at
configured densities for each mesh in parallel, and uploads the resulting
`.npz` files to an existing GCS bucket:

  gs://<DST_BUCKET>/<category_id>/<object_id>/pc_<N>.npz

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
    --densities 2048,10240,51200 \
    --jobs 16

Alternate (direct ZIP URI):
python scripts/extract_pointclouds.py \
    --project adlr2025 \
    --src_zip gs://shapenet_bucket/02747177.zip \
    --dst_bucket adlr2025-pointclouds \
    --densities 2048,10240,51200 \
    --jobs 16
"""
import argparse
import os
import subprocess
import tempfile
import glob
import numpy as np
import open3d as o3d
from multiprocessing import Pool, cpu_count


def run(cmd):
    """Run a shell command, raising on error."""
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True)


def sample_and_upload_mesh(args):
    mesh_file, category, densities, dst_bucket = args
    # Determine object_id as the parent of the folder containing the mesh
    # e.g., .../<category>/<object_id>/models/mesh.obj
    mesh_folder = os.path.dirname(mesh_file)
    object_dir = os.path.dirname(mesh_folder)
    object_id = os.path.basename(object_dir)

    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()

    with tempfile.TemporaryDirectory() as tmpm:
        for N in densities:
            pts = np.asarray(
                mesh.sample_points_uniformly(number_of_points=N).points
            )
            local_npz = os.path.join(tmpm, f"pc_{N}.npz")
            np.savez_compressed(local_npz, points=pts)
            gcs_path = f"gs://{dst_bucket}/{category}/{object_id}/pc_{N}.npz"
            run(f"gsutil -q cp {local_npz} {gcs_path}")
            print(f"Uploaded {gcs_path}")


def process_zip(zip_source, category, densities, dst_bucket, jobs):
    with tempfile.TemporaryDirectory() as tmp:
        zip_name = os.path.basename(zip_source)
        zip_path = os.path.join(tmp, zip_name)
        run(f"gsutil -q cp {zip_source} {zip_path}")

        # Unzip all .obj meshes preserving folder structure
        mesh_dir = os.path.join(tmp, category)
        os.makedirs(mesh_dir, exist_ok=True)
        run(f"unzip -q {zip_path} '*.obj' -d {mesh_dir}")

        mesh_files = glob.glob(os.path.join(mesh_dir, '**', '*.obj'), recursive=True)
        if not mesh_files:
            raise RuntimeError(f"No .obj files found in {zip_source}")

        work_items = [(mf, category, densities, dst_bucket) for mf in mesh_files]
        with Pool(processes=jobs) as pool:
            pool.map(sample_and_upload_mesh, work_items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", required=True, help="GCP project ID for gsutil and gcloud operations"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--src_bucket", help="GCS bucket name where category ZIPs live (no gs://)"
    )
    src.add_argument(
        "--src_zip", help="Full GCS URI to a category ZIP file"
    )
    parser.add_argument(
        "--category_id",
        help="Category ID (ZIP filename without .zip); required with --src_bucket"
    )
    parser.add_argument(
        "--dst_bucket", required=True,
        help="Destination GCS bucket for .npz outputs (no gs://)"
    )
    parser.add_argument(
        "--densities", type=lambda s: [int(x) for x in s.split(",")],
        default=[2048, 10240, 51200],
        help="Comma-separated list of point counts to sample, e.g. 2048,10240,51200"
    )
    parser.add_argument(
        "--jobs", type=int, default=cpu_count(),
        help="Number of parallel worker processes (default: CPU count)"
    )
    args = parser.parse_args()

    os.environ['GOOGLE_CLOUD_PROJECT'] = args.project

    if args.src_zip:
        zip_list = [args.src_zip]
        category = args.category_id or os.path.splitext(os.path.basename(args.src_zip))[0]
    else:
        if not args.category_id:
            parser.error("--category_id is required when using --src_bucket")
        zip_list = [f"gs://{args.src_bucket}/{args.category_id}.zip"]
        category = args.category_id

    for zs in zip_list:
        process_zip(zs, category, args.densities, args.dst_bucket, args.jobs)

if __name__ == '__main__':
    main()
