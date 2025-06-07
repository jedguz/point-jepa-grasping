# File: scripts/extract_pointclouds.py
"""
Script: extract_pointclouds.py
Location: project root under `scripts/`

Description:
Downloads ShapeNet category ZIPs from GCS, unpacks all `.obj` meshes,
samples point clouds at configured densities for each mesh *in parallel*,
and stages all `.npz` locally; then uses one bulk `gsutil -m cp` to push
everything to GCS, avoiding per-file network calls.

Output layout in bucket:
  gs://<DST_BUCKET>/<category_id>/<object_id>/pc_<N>.npz

Prerequisites:
1. Create bucket:
   gcloud storage buckets create gs://adlr2025-pointclouds \
     --project=adlr2025 --location=europe-west3 --storage-class=STANDARD
2. VM scopes include storage access.

Usage:
python scripts/extract_pointclouds.py \
    --project adlr2025 \
    --src_zip gs://shapenet_bucket/02747177.zip \
    --dst_bucket adlr2025-pointclouds \
    --densities 2048,10240,51200 \
    --jobs 16

Parallel across meshes, then one bulk upload per category.
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
    subprocess.check_call(cmd, shell=True)


def sample_mesh(args):
    mesh_file, densities = args
    mesh_id = os.path.basename(os.path.dirname(mesh_file))
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    outputs = []
    for N in densities:
        pts = np.asarray(mesh.sample_points_uniformly(number_of_points=N).points)
        out_file = f"{os.path.splitext(mesh_file)[0]}_pc_{N}.npz"
        np.savez_compressed(out_file, points=pts)
        outputs.append((out_file, mesh_id))
    return outputs


def process_category(zip_source, densities, dst_bucket, jobs):
    # Create temp dir and extract zip
    with tempfile.TemporaryDirectory() as tmp:
        run(f"gsutil -q cp {zip_source} {tmp}/archive.zip")
        run(f"unzip -q {tmp}/archive.zip -d {tmp}")
        # Find meshes
        mesh_files = glob.glob(os.path.join(tmp, '**', '*.obj'), recursive=True)
        if not mesh_files:
            raise RuntimeError("No .obj found in zip")
        # Parallel sample meshes
        work = [(mf, densities) for mf in mesh_files]
        with Pool(jobs) as p:
            results = p.map(sample_mesh, work)
        # Flatten results and build local staging dir
        staging = os.path.join(tmp, 'staging')
        for out_list in results:
            for file_path, mesh_id in out_list:
                dest_dir = os.path.join(staging, mesh_id)
                os.makedirs(dest_dir, exist_ok=True)
                os.rename(file_path, os.path.join(dest_dir, os.path.basename(file_path)))
        # Bulk upload
        run(f"gsutil -m cp -r {staging}/* gs://{dst_bucket}/{os.path.basename(zip_source).split('.')[0]}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--src_zip', required=True)
    parser.add_argument('--dst_bucket', required=True)
    parser.add_argument('--densities', type=lambda s: [int(x) for x in s.split(',')],
                        default=[2048,10240,51200])
    parser.add_argument('--jobs', type=int, default=cpu_count())
    args = parser.parse_args()
    os.environ['GOOGLE_CLOUD_PROJECT'] = args.project
    process_category(args.src_zip, args.densities, args.dst_bucket, args.jobs)

if __name__ == '__main__':
    main()
