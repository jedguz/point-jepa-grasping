#!/usr/bin/env python3
"""
ShapeNet Mesh to Point Cloud Converter
======================================

This script converts ShapeNet meshes (OBJ files) to point clouds using various sampling methods.
Supports batch processing of entire ShapeNet categories with multiple output formats.

Usage:
    python mesh_to_pointcloud.py
"""

import os
import numpy as np
import trimesh
import pandas as pd
import json
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import argparse
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input/Output directories
SHAPENET_DIR = "./shapenet_extracted"        # Where your extracted ShapeNet data is
POINTCLOUD_DIR = "./pointclouds"             # Where to save point clouds
METADATA_DIR = "./metadata"                  # Where to save processing metadata

# Point cloud generation settings
DEFAULT_NUM_POINTS = 2048                    # Default number of points to sample
POINT_CLOUD_FORMATS = ['npy', 'ply', 'txt'] # Output formats
SAMPLING_METHODS = ['surface', 'volume']     # surface = mesh surface, volume = solid interior

# Processing settings
MAX_WORKERS = multiprocessing.cpu_count()   # Number of parallel processes
BATCH_SIZE = 100                            # Process files in batches
SKIP_EXISTING = True                        # Skip already processed files

# Quality settings
MIN_VERTICES = 10                           # Skip meshes with too few vertices
MAX_VERTICES = 1000000                      # Skip meshes that are too large
REMOVE_DUPLICATES = True                    # Remove duplicate vertices
FIX_NORMALS = True                         # Attempt to fix mesh normals

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_directories():
    """Create necessary output directories"""
    dirs = [POINTCLOUD_DIR, METADATA_DIR]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def find_obj_files(shapenet_dir):
    """Find all OBJ files in the ShapeNet directory structure"""
    print(f"🔍 Scanning for OBJ files in {shapenet_dir}...")
    
    obj_files = []
    shapenet_path = Path(shapenet_dir)
    
    if not shapenet_path.exists():
        print(f"❌ ShapeNet directory not found: {shapenet_dir}")
        return []
    
    # Find all .obj files recursively
    for obj_file in shapenet_path.rglob("*.obj"):
        # Extract category and model ID from path
        parts = obj_file.parts
        
        # Typical ShapeNet structure: shapenet_extracted/category/model_id/models/model_normalized.obj
        if len(parts) >= 3:
            category = None
            model_id = None
            
            # Find category (8-digit number)
            for part in parts:
                if len(part) == 8 and part.isdigit():
                    category = part
                    break
            
            # Find model ID (32-character hash)
            for part in parts:
                if len(part) == 32:
                    model_id = part
                    break
            
            obj_files.append({
                'file_path': str(obj_file),
                'category': category,
                'model_id': model_id,
                'filename': obj_file.name,
                'size_mb': obj_file.stat().st_size / (1024 * 1024) if obj_file.exists() else 0
            })
    
    print(f"✓ Found {len(obj_files)} OBJ files")
    
    # Group by category
    categories = {}
    for obj in obj_files:
        cat = obj['category']
        if cat:
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(obj)
    
    print(f"📊 Categories found:")
    for cat, objs in categories.items():
        print(f"  {cat}: {len(objs)} models")
    
    return obj_files

def load_mesh(file_path):
    """Load a mesh file using trimesh"""
    try:
        mesh = trimesh.load(file_path, force='mesh')
        
        # Handle multiple meshes (trimesh sometimes returns a Scene)
        if hasattr(mesh, 'geometry'):
            # It's a Scene, get the first mesh
            if len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                return None
        
        return mesh
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

# =============================================================================
# POINT CLOUD GENERATION FUNCTIONS
# =============================================================================

def mesh_to_pointcloud_surface(mesh, num_points=DEFAULT_NUM_POINTS):
    """Sample points from mesh surface"""
    try:
        # Sample points from mesh surface
        points, face_indices = mesh.sample(num_points, return_index=True)
        
        # Get face normals for the sampled points
        if mesh.face_normals is not None and len(mesh.face_normals) > 0:
            normals = mesh.face_normals[face_indices]
        else:
            # Compute normals if not available
            try:
                mesh.compute_vertex_normals()
                normals = mesh.face_normals[face_indices] if mesh.face_normals is not None else None
            except:
                normals = None
        
        return points, normals
    except Exception as e:
        print(f"❌ Error sampling surface: {e}")
        return None, None

def mesh_to_pointcloud_volume(mesh, num_points=DEFAULT_NUM_POINTS):
    """Sample points from mesh volume (solid interior)"""
    try:
        # Check if mesh is watertight
        if not mesh.is_watertight:
            print("⚠️  Mesh is not watertight, using surface sampling instead")
            return mesh_to_pointcloud_surface(mesh, num_points)
        
        # Sample points from volume
        points = mesh.sample_volume(num_points)
        
        # For volume sampling, normals are not well-defined
        normals = None
        
        return points, normals
    except Exception as e:
        print(f"❌ Error sampling volume: {e}")
        # Fallback to surface sampling
        return mesh_to_pointcloud_surface(mesh, num_points)

def preprocess_mesh(mesh):
    """Preprocess mesh before point cloud generation"""
    if mesh is None:
        return None
    
    try:
        # Remove duplicate vertices
        if REMOVE_DUPLICATES:
            mesh.remove_duplicate_faces()
            mesh.remove_unreferenced_vertices()
        
        # Fix normals
        if FIX_NORMALS:
            try:
                mesh.fix_normals()
            except:
                pass  # Some meshes can't have normals fixed
        
        # Check mesh quality
        if len(mesh.vertices) < MIN_VERTICES:
            print(f"❌ Mesh has too few vertices: {len(mesh.vertices)}")
            return None
        
        if len(mesh.vertices) > MAX_VERTICES:
            print(f"❌ Mesh has too many vertices: {len(mesh.vertices)}")
            return None
        
        return mesh
    except Exception as e:
        print(f"❌ Error preprocessing mesh: {e}")
        return None

# =============================================================================
# FILE I/O FUNCTIONS
# =============================================================================

def save_pointcloud_npy(points, normals, output_path):
    """Save point cloud as NumPy array"""
    try:
        if normals is not None:
            # Combine points and normals
            data = np.hstack([points, normals])
        else:
            data = points
        
        np.save(output_path, data)
        return True
    except Exception as e:
        print(f"❌ Error saving NPY: {e}")
        return False

def save_pointcloud_ply(points, normals, output_path):
    """Save point cloud as PLY file"""
    try:
        # Create a point cloud using trimesh
        if normals is not None:
            pointcloud = trimesh.PointCloud(vertices=points, vertex_normals=normals)
        else:
            pointcloud = trimesh.PointCloud(vertices=points)
        
        pointcloud.export(output_path)
        return True
    except Exception as e:
        print(f"❌ Error saving PLY: {e}")
        return False

def save_pointcloud_txt(points, normals, output_path):
    """Save point cloud as text file"""
    try:
        if normals is not None:
            # Format: x y z nx ny nz
            data = np.hstack([points, normals])
            header = "x y z nx ny nz"
        else:
            # Format: x y z
            data = points
            header = "x y z"
        
        np.savetxt(output_path, data, header=header, comments='# ')
        return True
    except Exception as e:
        print(f"❌ Error saving TXT: {e}")
        return False

def save_pointcloud(points, normals, base_path, formats):
    """Save point cloud in multiple formats"""
    saved_files = []
    
    for fmt in formats:
        output_path = f"{base_path}.{fmt}"
        
        if fmt == 'npy':
            success = save_pointcloud_npy(points, normals, output_path)
        elif fmt == 'ply':
            success = save_pointcloud_ply(points, normals, output_path)
        elif fmt == 'txt':
            success = save_pointcloud_txt(points, normals, output_path)
        else:
            print(f"❌ Unknown format: {fmt}")
            continue
        
        if success:
            saved_files.append(output_path)
    
    return saved_files

# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def process_single_mesh(args):
    """Process a single mesh file (for parallel processing)"""
    obj_info, num_points, sampling_methods, output_formats = args
    
    file_path = obj_info['file_path']
    category = obj_info['category']
    model_id = obj_info['model_id']
    
    print(f"🔄 Processing: {category}/{model_id}")
    
    results = {
        'file_path': file_path,
        'category': category,
        'model_id': model_id,
        'success': False,
        'error': None,
        'output_files': [],
        'mesh_stats': {},
        'processing_time': 0
    }
    
    start_time = time.time()
    
    try:
        # Load mesh
        mesh = load_mesh(file_path)
        if mesh is None:
            results['error'] = "Failed to load mesh"
            return results
        
        # Preprocess mesh
        mesh = preprocess_mesh(mesh)
        if mesh is None:
            results['error'] = "Mesh failed preprocessing"
            return results
        
        # Record mesh statistics
        results['mesh_stats'] = {
            'vertices': len(mesh.vertices),
            'faces': len(mesh.faces),
            'watertight': mesh.is_watertight,
            'volume': float(mesh.volume) if mesh.is_watertight else None,
            'surface_area': float(mesh.area),
            'bounds': mesh.bounds.tolist()
        }
        
        # Create output directory structure
        category_dir = os.path.join(POINTCLOUD_DIR, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Process each sampling method
        for method in sampling_methods:
            # Generate point cloud
            if method == 'surface':
                points, normals = mesh_to_pointcloud_surface(mesh, num_points)
            elif method == 'volume':
                points, normals = mesh_to_pointcloud_volume(mesh, num_points)
            else:
                print(f"❌ Unknown sampling method: {method}")
                continue
            
            if points is None:
                continue
            
            # Create output filename
            base_name = f"{model_id}_{method}_{num_points}"
            base_path = os.path.join(category_dir, base_name)
            
            # Skip if files already exist
            if SKIP_EXISTING:
                existing_files = [f"{base_path}.{fmt}" for fmt in output_formats]
                if all(os.path.exists(f) for f in existing_files):
                    print(f"⏭️  Skipping {base_name} (already exists)")
                    results['output_files'].extend(existing_files)
                    continue
            
            # Save point cloud
            saved_files = save_pointcloud(points, normals, base_path, output_formats)
            results['output_files'].extend(saved_files)
        
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Error processing {file_path}: {e}")
    
    results['processing_time'] = time.time() - start_time
    return results

def process_batch(obj_files, num_points=DEFAULT_NUM_POINTS, 
                 sampling_methods=SAMPLING_METHODS, 
                 output_formats=POINT_CLOUD_FORMATS,
                 max_workers=MAX_WORKERS):
    """Process a batch of mesh files"""
    
    print(f"\n🚀 Starting batch processing of {len(obj_files)} files")
    print(f"   Workers: {max_workers}")
    print(f"   Points per cloud: {num_points}")
    print(f"   Sampling methods: {sampling_methods}")
    print(f"   Output formats: {output_formats}")
    print("-" * 60)
    
    # Prepare arguments for parallel processing
    args_list = [
        (obj_info, num_points, sampling_methods, output_formats)
        for obj_info in obj_files
    ]
    
    results = []
    successful = 0
    failed = 0
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_mesh, args) for args in args_list]
        
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                results.append(result)
                
                if result['success']:
                    successful += 1
                    print(f"✓ {result['category']}/{result['model_id']} "
                          f"({result['processing_time']:.1f}s)")
                else:
                    failed += 1
                    print(f"❌ {result['category']}/{result['model_id']}: {result['error']}")
                
                # Progress update
                if (i + 1) % 10 == 0:
                    print(f"📊 Progress: {i + 1}/{len(obj_files)} "
                          f"(✓{successful} ❌{failed})")
                    
            except Exception as e:
                failed += 1
                print(f"❌ Processing error: {e}")
    
    return results, successful, failed

# =============================================================================
# ANALYSIS AND REPORTING
# =============================================================================

def analyze_results(results):
    """Analyze processing results and generate statistics"""
    print(f"\n📊 PROCESSING ANALYSIS")
    print("=" * 60)
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"Total files processed: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    print(f"Success rate: {len(successful_results)/len(results)*100:.1f}%")
    
    if successful_results:
        # Processing time statistics
        times = [r['processing_time'] for r in successful_results]
        print(f"\nProcessing time statistics:")
        print(f"  Average: {np.mean(times):.2f}s")
        print(f"  Median: {np.median(times):.2f}s")
        print(f"  Min: {np.min(times):.2f}s")
        print(f"  Max: {np.max(times):.2f}s")
        
        # Mesh statistics
        vertex_counts = [r['mesh_stats']['vertices'] for r in successful_results if 'vertices' in r['mesh_stats']]
        if vertex_counts:
            print(f"\nMesh complexity statistics:")
            print(f"  Average vertices: {np.mean(vertex_counts):.0f}")
            print(f"  Median vertices: {np.median(vertex_counts):.0f}")
            print(f"  Min vertices: {np.min(vertex_counts):.0f}")
            print(f"  Max vertices: {np.max(vertex_counts):.0f}")
        
        # Category breakdown
        categories = {}
        for result in successful_results:
            cat = result['category']
            if cat:
                categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\nCategory breakdown:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count} models")
    
    # Error analysis
    if failed_results:
        print(f"\nError analysis:")
        error_types = {}
        for result in failed_results:
            error = result['error'] or 'Unknown error'
            error_types[error] = error_types.get(error, 0) + 1
        
        for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error}: {count} occurrences")
    
    return {
        'total': len(results),
        'successful': len(successful_results),
        'failed': len(failed_results),
        'success_rate': len(successful_results)/len(results) if results else 0,
        'categories': categories if successful_results else {},
        'error_types': error_types if failed_results else {}
    }

def save_processing_report(results, analysis, output_path):
    """Save detailed processing report"""
    report = {
        'summary': analysis,
        'processing_details': results,
        'configuration': {
            'num_points': DEFAULT_NUM_POINTS,
            'sampling_methods': SAMPLING_METHODS,
            'output_formats': POINT_CLOUD_FORMATS,
            'max_workers': MAX_WORKERS
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"📄 Processing report saved to: {output_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function"""
    print("=" * 60)
    print("🎯 SHAPENET MESH TO POINT CLOUD CONVERTER")
    print("=" * 60)
    
    # Setup
    setup_directories()
    
    # Find OBJ files
    obj_files = find_obj_files(SHAPENET_DIR)
    if not obj_files:
        print("❌ No OBJ files found!")
        return
    
    # User options
    print(f"\n⚙️  Configuration:")
    print(f"   Input directory: {SHAPENET_DIR}")
    print(f"   Output directory: {POINTCLOUD_DIR}")
    print(f"   Found models: {len(obj_files)}")
    print(f"   Points per cloud: {DEFAULT_NUM_POINTS}")
    print(f"   Sampling methods: {SAMPLING_METHODS}")
    print(f"   Output formats: {POINT_CLOUD_FORMATS}")
    print(f"   Max workers: {MAX_WORKERS}")
    
    # Confirm processing
    proceed = input(f"\n❓ Process {len(obj_files)} models? (y/n): ").strip().lower()
    if proceed != 'y':
        print("❌ Processing cancelled")
        return
    
    # Process files
    start_time = time.time()
    results, successful, failed = process_batch(obj_files)
    total_time = time.time() - start_time
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Save report
    report_path = os.path.join(METADATA_DIR, f"processing_report_{int(time.time())}.json")
    save_processing_report(results, analysis, report_path)
    
    # Final summary
    print(f"\n🎉 PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📊 Processed: {successful} successful, {failed} failed")
    print(f"📁 Output directory: {POINTCLOUD_DIR}")
    print(f"📄 Report: {report_path}")
    
    if successful > 0:
        total_files = successful * len(SAMPLING_METHODS) * len(POINT_CLOUD_FORMATS)
        print(f"🎯 Generated ~{total_files} point cloud files")

if __name__ == "__main__":
    main()