# File: scripts/extract_pointclouds.py
"""
Script: extract_pointclouds.py
Location: place this file in the project root under the `scripts/` directory.

Description:
Downloads a ShapeNet category ZIP from your GCS bucket, unpacks all
`.obj` meshes (preserving subdirectory structure), samples point clouds at
configured densities for each mesh, and uploads the resulting `.npz` files to
your existing GCS bucket (bucket must be created beforehand):

  gs://<DST_BUCKET>/<category_id>/<mesh_id>/pc_<N>.npz

Prerequisite:
 1. Ensure the destination bucket `adlr2025-pointclouds` exists:
      gcloud storage buckets create gs://adlr2025-pointclouds \
        --project=adlr2025 --location=europe-west3 --storage-class=STANDARD
 2. Make sure your VM or environment has proper GCS write scopes or credentials:
    - If on a Compute Engine VM, recreate it with:
      `--scopes=https://www.googleapis.com/auth/cloud-platform`
    - Or authenticate locally via `gcloud auth login` or a service account with Storage Object Admin.

Usage:
python scripts/extract_pointclouds.py \
    --project adlr2025 \
    --src_bucket shapenet_bucket \
    --category_id 02747177 \
    --dst_bucket adlr2025-pointclouds \
    --densities 2048 10240 51200

Alternate (direct ZIP URI):
python scripts/extract_pointclouds.py \
    --project adlr2025 \
    --src_zip gs://shapenet_bucket/02747177.zip \
    --dst_bucket adlr2025-pointclouds \
    --densities 2048 10240 51200
"""
import argparse
import os
import subprocess
import tempfile
import glob
import open3d as o3d
import numpy as np


def run(cmd):
    """Run a shell command, raising on error."""
    print(f"$ {cmd}")
    subprocess.check_call(cmd, shell=True)


def process_zip(zip_source, category, densities, dst_bucket):
    with tempfile.TemporaryDirectory() as tmp:
        # Download ZIP
        zip_name = os.path.basename(zip_source)
        zip_path = os.path.join(tmp, zip_name)
        run(f"gsutil -q cp {zip_source} {zip_path}")

        # Unzip all .obj meshes preserving folder structure
        mesh_dir = os.path.join(tmp, category)
        os.makedirs(mesh_dir, exist_ok=True)
        run(f"unzip -q {zip_path} '*.obj' -d {mesh_dir}")

        # Find all .obj files recursively
        mesh_files = glob.glob(os.path.join(mesh_dir, '**', '*.obj'), recursive=True)
        if not mesh_files:
            raise RuntimeError(f"No .obj files found in {zip_source}")

        # Sample and upload per mesh
        for mesh_file in mesh_files:
            mesh_id = os.path.basename(os.path.dirname(mesh_file))
            mesh = o3d.io.read_triangle_mesh(mesh_file)
            mesh.compute_vertex_normals()

            for N in densities:
                pts = np.asarray(
                    mesh.sample_points_uniformly(number_of_points=N).points
                )
                # Save locally then upload
                npz_tmp = os.path.join(tmp, f"{category}_{mesh_id}_pc_{N}.npz")
                np.savez_compressed(npz_tmp, points=pts)
                gcs_path = f"gs://{dst_bucket}/{category}/{mesh_id}/pc_{N}.npz"
                run(f"gsutil -q cp {npz_tmp} {gcs_path}")
                print(f"Uploaded {gcs_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project", required=True,
        help="GCP project ID for gsutil operations"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--src_bucket", help="GCS bucket with zips (no gs://)")
    src.add_argument("--src_zip", help="Full GCS URI to a ZIP file")
    parser.add_argument(
        "--category_id",
        help="Category ID (zip filename without .zip); required with --src_bucket"
    )
    parser.add_argument(
        "--dst_bucket", required=True,
        help="Destination GCS bucket for .npz outputs (no gs://)"
    )
    parser.add_argument(
        "--densities", nargs='+', type=int,
        default=[2048, 10240, 51200],
        help="Point counts to sample (e.g. 2048 10240 51200)"
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
            dst_bucket=args.dst_bucket
        )

if __name__ == '__main__':
    main()
