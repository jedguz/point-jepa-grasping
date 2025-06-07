# File: scripts/extract_pointclouds.py
"""
Script: extract_pointclouds.py
Location: project root under `scripts/`

Description:
Downloads ShapeNet category ZIPs from GCS, unpacks all `.obj` meshes,
samples point clouds at configured densities for each mesh *in parallel*,
and stages all `.npz` locally; then uses one bulk `gsutil -m cp` to push
everything to GCS in one shot per category.

Output layout in bucket:
  gs://<DST_BUCKET>/<category_id>/<mesh_id>/pc_<N>.npz

Prerequisites:
1. Create your bucket:
   gcloud storage buckets create gs://adlr2025-pointclouds \
     --project=adlr2025 --location=europe-west3 --storage-class=STANDARD
2. VM scopes include storage access: `--scopes=https://www.googleapis.com/auth/cloud-platform`

Usage:
python scripts/extract_pointclouds.py \
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
    """Run a shell command, raising on failure."""
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)


def sample_mesh(mesh_file, densities, tmp_dir):
    """
    Sample a single mesh at given densities.
    Returns list of (local_output_path, mesh_id).
    """
    mesh_id = os.path.splitext(os.path.basename(mesh_file))[0]
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    outputs = []
    for N in densities:
        pts = np.asarray(mesh.sample_points_uniformly(number_of_points=N).points)
        out_dir = os.path.join(tmp_dir, mesh_id)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"pc_{N}.npz")
        np.savez_compressed(out_file, points=pts)
        outputs.append(out_file)
    return outputs


def process_category(zip_source, densities, dst_bucket, jobs):
    """
    Process one zip: download, extract, sample all meshes, and bulk upload.
    """
    category_id = os.path.splitext(os.path.basename(zip_source))[0]
    with tempfile.TemporaryDirectory() as tmp:
        # download and extract
        archive = os.path.join(tmp, 'archive.zip')
        run(f"gsutil -q cp {zip_source} {archive}")
        run(f"unzip -q {archive} -d {tmp}")
        # find all meshes
        mesh_files = glob.glob(os.path.join(tmp, '**', '*.obj'), recursive=True)
        if not mesh_files:
            raise RuntimeError(f"No .obj files found in {zip_source}")
        # sample in parallel
        print(f"Sampling {len(mesh_files)} meshes in category {category_id} with {jobs} workers...")
        with Pool(processes=jobs) as pool:
            # prepare args for top‐level sample_mesh (which is picklable)
            args_list = [(mf, densities, tmp) for mf in mesh_files]
            all_outputs = pool.starmap(sample_mesh, args_list)
        # bulk upload staging directory
        staging_dir = tmp  # contains subdirs per mesh
        dest = f"gs://{dst_bucket}/{category_id}/"
        print(f"Uploading all sampled pointclouds for {category_id} to {dest}...")
        run(f"gsutil -m cp -r {staging_dir}/* {dest}")
        print(f"Category {category_id} done.")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--src_zip', type=str,
                       help='GCS URI of the category zip (e.g. gs://bucket/02747177.zip)')
    parser.add_argument('--dst_bucket', type=str, required=True,
                        help='Destination GCS bucket (just name, e.g. adlr2025-pointclouds)')
    parser.add_argument('--densities', type=lambda s: [int(x) for x in s.split(',')],
                        default=[2048,10240,51200],
                        help='Comma-separated list of point counts')
    parser.add_argument('--jobs', type=int, default=cpu_count(),
                        help='Number of parallel workers (defaults to all CPUs)')

    args = parser.parse_args()
    process_category(args.src_zip, args.densities, args.dst_bucket, args.jobs)


if __name__ == '__main__':
    main()