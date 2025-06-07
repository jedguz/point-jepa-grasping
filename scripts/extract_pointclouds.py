#!/usr/bin/env python3
"""
Updated extract_pointclouds.py
- Default densities to 2048,10240,25000
- Silence Open3D INFO logs
- Show progress with tqdm
- Default parallel workers to half of CPUs
"""
import argparse
import os
import subprocess
import tempfile
import glob
import numpy as np
import open3d as o3d
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Silence verbose Open3D INFO messages
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)

def run(cmd):
    """Run a shell command, raising on failure."""
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)


def sample_mesh(mesh_file, densities, tmp_dir):
    """
    Sample a single mesh at given densities. Returns list of output file paths.
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


def _sample_wrapper(args):
    # top-level wrapper so Pool can pickle
    return sample_mesh(*args)


def process_category(zip_source, densities, dst_bucket, jobs):
    category_id = os.path.splitext(os.path.basename(zip_source))[0]
    with tempfile.TemporaryDirectory() as tmp:
        # download and unpack
        archive = os.path.join(tmp, 'archive.zip')
        run(f"gsutil -q cp {zip_source} {archive}")
        run(f"unzip -q {archive} -d {tmp}")

        # find all .obj files
        mesh_files = glob.glob(os.path.join(tmp, '**', '*.obj'), recursive=True)
        if not mesh_files:
            raise RuntimeError(f"No .obj files found in {zip_source}")

        print(f"Sampling {len(mesh_files)} meshes in category {category_id} with {jobs} workers...")
        args_list = [(mf, densities, tmp) for mf in mesh_files]

        # parallel sample with progress bar
        all_outputs = []
        with Pool(processes=jobs) as pool:
            for outputs in tqdm(pool.imap_unordered(_sample_wrapper, args_list), total=len(mesh_files)):
                all_outputs.extend(outputs)

        # upload everything at once
        dest = f"gs://{dst_bucket}/{category_id}/"
        print(f"Uploading {len(all_outputs)} files for {category_id} to {dest}...")
        run(f"gsutil -m cp -r {tmp}/* {dest}")
        print(f"Category {category_id} done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract and sample point clouds from ShapeNet category ZIPs with progress and tuned defaults."
    )
    parser.add_argument(
        '--src_zip', type=str, required=True,
        help='GCS URI of the category zip (e.g. gs://bucket/02747177.zip)'
    )
    parser.add_argument(
        '--dst_bucket', type=str, required=True,
        help='Destination GCS bucket name (e.g. adlr2025-pointclouds)'
    )
    parser.add_argument(
        '--densities', type=lambda s: [int(x) for x in s.split(',')],
        default=[2048,10240,25000],
        help='Comma-separated point counts (default: 2048,10240,25000)'
    )
    parser.add_argument(
        '--jobs', type=int, default=max(1, cpu_count() // 2),
        help='Number of parallel workers (default: half of available CPUs)'
    )
    args = parser.parse_args()
    process_category(args.src_zip, args.densities, args.dst_bucket, args.jobs)
