#!/usr/bin/env python3
"""
ShapeNet Core GCS Loader for GCE VM Projects
============================================

This script downloads and extracts all 55 ShapeNet Core zip files from 
Google Cloud Storage to your GCE VM project.

Usage:
    1. Update the configuration variables below
    2. Run the script: python shapenet_loader.py
"""

import os
import zipfile
import pandas as pd
from google.cloud import storage
from google.auth import default
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# =============================================================================
# CONFIGURATION - UPDATE THESE VALUES FOR YOUR PROJECT
# =============================================================================

PROJECT_ID = "your-project-id"           # Replace with your GCP project ID
BUCKET_NAME = "your-bucket-name"         # Replace with your GCS bucket name
LOCAL_DATA_DIR = "./shapenet_data"       # Local directory for ShapeNet data
EXTRACT_DIR = "./shapenet_extracted"     # Directory for extracted models
MAX_WORKERS = 2                          # Number of parallel downloads/extractions (reduced for stability)

# Performance tuning options
EXTRACTION_BATCH_SIZE = 5                # How many zips to extract simultaneously
PROGRESS_INTERVAL = 5000                 # Show progress every N files during extraction
USE_SSD_OPTIMIZATIONS = True             # Enable optimizations for SSD storage

# Category filtering - Set to None to extract all, or specify categories to extract
CATEGORIES_TO_EXTRACT = None             # Extract all categories
# CATEGORIES_TO_EXTRACT = ["02691156", "02958343", "03001627"]  # Only airplane, car, chair
# CATEGORIES_TO_EXTRACT = ["02691156", "02773838", "02801938", "02808440", "02818832"]  # First 5 categories

# ShapeNet Core typically has these categories (you can modify based on your bucket structure)
SHAPENET_CATEGORIES = [
    "02691156",  # airplane
    "02773838",  # bag
    "02801938",  # basket
    "02808440",  # bathtub
    "02818832",  # bed
    "02828884",  # bench
    "02876657",  # bottle
    "02880940",  # bowl
    "02924116",  # bus
    "02933112",  # cabinet
    "02747177",  # trash can
    "02942699",  # camera
    "02954340",  # cap
    "02958343",  # car
    "03001627",  # chair
    "03046257",  # clock
    "03207941",  # dishwasher
    "03211117",  # display
    "03261776",  # earphone
    "03325088",  # faucet
    "03337140",  # file cabinet
    "03467517",  # guitar
    "03513137",  # helmet
    "03593526",  # jar
    "03624134",  # knife
    "03636649",  # lamp
    "03642806",  # laptop
    "03691459",  # loudspeaker
    "03710193",  # mailbox
    "03759954",  # microphone
    "03761084",  # microwave
    "03790512",  # motorbike
    "03797390",  # mug
    "03928116",  # piano
    "03938244",  # pillow
    "03948459",  # pistol
    "03991062",  # pot
    "04004475",  # printer
    "04074963",  # remote control
    "04090263",  # rifle
    "04099429",  # rocket
    "04225987",  # skateboard
    "04256520",  # sofa
    "04330267",  # stove
    "04379243",  # table
    "04401088",  # telephone
    "04460130",  # tower
    "04468005",  # train
    "04530566",  # vessel
    "04554684",  # washer
    "02992529",  # cellphone
    "03085013",  # keyboard
    "03366839",  # folder
    "04401088",  # phone
]

# =============================================================================
# SETUP AND AUTHENTICATION
# =============================================================================

def setup_environment():
    """Set up the local environment and authenticate with Google Cloud"""
    
    # Create directories
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    
    print("🚀 ShapeNet Core Loader Starting...")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Bucket Name: {BUCKET_NAME}")
    print(f"Download Directory: {LOCAL_DATA_DIR}")
    print(f"Extract Directory: {EXTRACT_DIR}")
    print(f"Max Workers: {MAX_WORKERS}")
    print("-" * 60)
    
    # Authenticate with Google Cloud
    try:
        credentials, project = default()
        client = storage.Client(project=PROJECT_ID, credentials=credentials)
        print("✓ Successfully authenticated using default credentials")
        print(f"✓ Using project: {project}")
        return client
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("💡 Make sure your GCE VM has the proper service account permissions")
        return None

def connect_to_bucket(client, bucket_name):
    """Connect to the specified GCS bucket"""
    try:
        bucket = client.bucket(bucket_name)
        
        if bucket.exists():
            print(f"✓ Successfully connected to bucket: {bucket_name}")
            bucket.reload()
            print(f"✓ Bucket location: {bucket.location}")
            return bucket
        else:
            print(f"❌ Bucket {bucket_name} does not exist or is not accessible")
            return None
            
    except Exception as e:
        print(f"❌ Error connecting to bucket: {e}")
        return None

# =============================================================================
# SHAPENET DISCOVERY FUNCTIONS
# =============================================================================

def discover_shapenet_zips(bucket, prefix=""):
    """Discover all zip files in the bucket that contain ShapeNet data"""
    print(f"\n🔍 Discovering ShapeNet zip files...")
    
    blobs = list(bucket.list_blobs(prefix=prefix))
    zip_files = []
    
    for blob in blobs:
        if blob.name.lower().endswith('.zip'):
            size_mb = blob.size / (1024 * 1024) if blob.size else 0
            zip_files.append({
                'name': blob.name,
                'size_mb': round(size_mb, 2),
                'updated': blob.updated,
                'blob': blob
            })
    
    print(f"📦 Found {len(zip_files)} zip files:")
    print("-" * 80)
    
    total_size = 0
    for zf in zip_files:
        print(f"📄 {zf['name']} ({zf['size_mb']:.2f} MB)")
        total_size += zf['size_mb']
    
    print(f"\n📊 Total size: {total_size:.2f} MB ({total_size/1024:.2f} GB)")
    
    return zip_files

def filter_zip_files_by_category(zip_files, categories_to_extract=None):
    """Filter zip files to only include specified categories"""
    if categories_to_extract is None:
        print("📦 Processing all categories")
        return zip_files
    
    print(f"🎯 Filtering for categories: {categories_to_extract}")
    
    filtered_files = []
    for zip_info in zip_files:
        # Check if any of the specified categories is in the filename
        for category in categories_to_extract:
            if category in zip_info['name']:
                filtered_files.append(zip_info)
                print(f"  ✓ Including: {zip_info['name']}")
                break
    
    excluded_count = len(zip_files) - len(filtered_files)
    print(f"📊 Filtered result: {len(filtered_files)} files selected, {excluded_count} excluded")
    
    return filtered_files

def categorize_zip_files(zip_files):
    """Try to categorize zip files by ShapeNet category"""
    categorized = {}
    uncategorized = []
    
    for zf in zip_files:
        category_found = False
        for category in SHAPENET_CATEGORIES:
            if category in zf['name']:
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(zf)
                category_found = True
                break
        
        if not category_found:
            uncategorized.append(zf)
    
    print(f"\n📋 Categorization Summary:")
    print(f"  • Categorized: {len(categorized)} categories")
    print(f"  • Uncategorized: {len(uncategorized)} files")
    
    return categorized, uncategorized
    """Try to categorize zip files by ShapeNet category"""
    categorized = {}
    uncategorized = []
    
    for zf in zip_files:
        category_found = False
        for category in SHAPENET_CATEGORIES:
            if category in zf['name']:
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(zf)
                category_found = True
                break
        
        if not category_found:
            uncategorized.append(zf)
    
    print(f"\n📋 Categorization Summary:")
    print(f"  • Categorized: {len(categorized)} categories")
    print(f"  • Uncategorized: {len(uncategorized)} files")
    
    return categorized, uncategorized

# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_zip_file(bucket, zip_info, local_dir, progress_callback=None):
    """Download a single zip file"""
    blob_name = zip_info['name']
    local_path = os.path.join(local_dir, os.path.basename(blob_name))
    
    try:
        blob = bucket.blob(blob_name)
        
        # Check if file already exists
        if os.path.exists(local_path):
            local_size = os.path.getsize(local_path)
            remote_size = blob.size
            if local_size == remote_size:
                print(f"⏭️  Skipping {blob_name} (already exists)")
                return local_path, True
        
        print(f"⬇️  Downloading {blob_name} ({zip_info['size_mb']:.2f} MB)...")
        start_time = time.time()
        
        blob.download_to_filename(local_path)
        
        elapsed = time.time() - start_time
        speed = zip_info['size_mb'] / elapsed if elapsed > 0 else 0
        
        print(f"✓ Downloaded {blob_name} in {elapsed:.1f}s ({speed:.1f} MB/s)")
        
        if progress_callback:
            progress_callback(blob_name, True)
            
        return local_path, True
        
    except Exception as e:
        print(f"❌ Error downloading {blob_name}: {e}")
        if progress_callback:
            progress_callback(blob_name, False)
        return None, False

def download_all_zips(bucket, zip_files, local_dir, max_workers=MAX_WORKERS):
    """Download all zip files using parallel workers"""
    print(f"\n⬇️  Starting download of {len(zip_files)} files using {max_workers} workers...")
    
    downloaded_files = []
    failed_downloads = []
    
    # Progress tracking
    completed = threading.Event()
    progress_lock = threading.Lock()
    progress_data = {'completed': 0, 'total': len(zip_files), 'failed': 0}
    
    def progress_callback(filename, success):
        with progress_lock:
            progress_data['completed'] += 1
            if not success:
                progress_data['failed'] += 1
            print(f"📊 Progress: {progress_data['completed']}/{progress_data['total']} "
                  f"(Failed: {progress_data['failed']})")
    
    # Use ThreadPoolExecutor for parallel downloads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_zip = {
            executor.submit(download_zip_file, bucket, zf, local_dir, progress_callback): zf 
            for zf in zip_files
        }
        
        # Collect results
        for future in as_completed(future_to_zip):
            zip_info = future_to_zip[future]
            try:
                local_path, success = future.result()
                if success and local_path:
                    downloaded_files.append(local_path)
                else:
                    failed_downloads.append(zip_info['name'])
            except Exception as e:
                print(f"❌ Download failed for {zip_info['name']}: {e}")
                failed_downloads.append(zip_info['name'])
    
    print(f"\n📦 Download Summary:")
    print(f"  ✓ Successfully downloaded: {len(downloaded_files)}")
    print(f"  ❌ Failed downloads: {len(failed_downloads)}")
    
    if failed_downloads:
        print(f"  Failed files: {failed_downloads}")
    
    return downloaded_files, failed_downloads

# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_zip_file(zip_path, extract_dir, preserve_structure=True, progress_interval=5000):
    """Extract a single zip file with optimized performance"""
    try:
        zip_name = os.path.basename(zip_path)
        print(f"📂 Extracting {zip_name}...")
        
        # Create extraction subdirectory if preserving structure
        if preserve_structure:
            extract_subdir = os.path.join(extract_dir, os.path.splitext(zip_name)[0])
            os.makedirs(extract_subdir, exist_ok=True)
            final_extract_dir = extract_subdir
        else:
            final_extract_dir = extract_dir
        
        start_time = time.time()
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get info about the zip
            file_list = zip_ref.infolist()
            file_count = len(file_list)
            
            print(f"  📄 {file_count} files to extract...")
            
            # Extract files with progress updates for large archives
            if file_count > 1000:
                extracted = 0
                last_update = 0
                
                for file_info in file_list:
                    # Skip directories
                    if file_info.is_dir():
                        continue
                        
                    # Extract individual file
                    zip_ref.extract(file_info, final_extract_dir)
                    extracted += 1
                    
                    # Progress update every N files
                    if extracted - last_update >= progress_interval:
                        percent = (extracted / file_count) * 100
                        elapsed_so_far = time.time() - start_time
                        rate = extracted / elapsed_so_far if elapsed_so_far > 0 else 0
                        eta = (file_count - extracted) / rate if rate > 0 else 0
                        
                        print(f"    Progress: {extracted}/{file_count} ({percent:.1f}%) "
                              f"[{rate:.0f} files/s, ETA: {eta/60:.1f}m]")
                        last_update = extracted
            else:
                # For smaller archives, extract normally
                zip_ref.extractall(final_extract_dir)
        
        elapsed = time.time() - start_time
        rate = file_count / elapsed if elapsed > 0 else 0
        print(f"✓ Extracted {zip_name} ({file_count} files) in {elapsed:.1f}s ({rate:.0f} files/s)")
        
        return final_extract_dir, file_count, True
        
    except Exception as e:
        print(f"❌ Error extracting {zip_path}: {e}")
        return None, 0, False

def extract_all_zips(zip_files, extract_dir, max_workers=EXTRACTION_BATCH_SIZE, preserve_structure=True):
    """Extract all zip files using parallel workers with performance optimizations"""
    print(f"\n📂 Starting extraction of {len(zip_files)} files...")
    print(f"   Using {max_workers} parallel extractions")
    print(f"   Progress updates every {PROGRESS_INTERVAL} files")
    
    extracted_dirs = []
    failed_extractions = []
    total_files = 0
    
    # Sort zip files by size (smallest first for better load balancing)
    zip_files_sorted = sorted(zip_files, key=lambda x: os.path.getsize(x) if os.path.exists(x) else 0)
    
    # Progress tracking
    progress_lock = threading.Lock()
    progress_data = {'completed': 0, 'total': len(zip_files), 'failed': 0}
    
    def extract_with_progress(zip_path):
        result_dir, file_count, success = extract_zip_file(
            zip_path, extract_dir, preserve_structure, PROGRESS_INTERVAL
        )
        
        with progress_lock:
            progress_data['completed'] += 1
            if not success:
                progress_data['failed'] += 1
            
            remaining = progress_data['total'] - progress_data['completed']
            print(f"📊 Overall Progress: {progress_data['completed']}/{progress_data['total']} "
                  f"({remaining} remaining, {progress_data['failed']} failed)")
        
        return result_dir, file_count, success, zip_path
    
    # Use ThreadPoolExecutor with reduced workers for extraction
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all extraction tasks
        futures = [executor.submit(extract_with_progress, zip_path) for zip_path in zip_files_sorted]
        
        # Collect results
        for future in as_completed(futures):
            try:
                result_dir, file_count, success, zip_path = future.result()
                if success:
                    extracted_dirs.append(result_dir)
                    total_files += file_count
                else:
                    failed_extractions.append(zip_path)
            except Exception as e:
                print(f"❌ Extraction failed: {e}")
                failed_extractions.append("unknown")
    
    print(f"\n📁 Extraction Summary:")
    print(f"  ✓ Successfully extracted: {len(extracted_dirs)} archives")
    print(f"  📄 Total files extracted: {total_files:,}")
    print(f"  ❌ Failed extractions: {len(failed_extractions)}")
    
    return extracted_dirs, failed_extractions, total_files

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_shapenet_structure(extract_dir):
    """Analyze the structure of extracted ShapeNet data"""
    print(f"\n🔍 Analyzing ShapeNet data structure in {extract_dir}...")
    
    structure_info = {}
    total_models = 0
    
    extract_path = Path(extract_dir)
    
    for category_dir in extract_path.iterdir():
        if category_dir.is_dir():
            category_name = category_dir.name
            
            # Count models in this category
            model_count = 0
            model_dirs = []
            
            for item in category_dir.rglob('*'):
                if item.is_dir() and len(item.name) == 32:  # ShapeNet model IDs are 32 chars
                    model_count += 1
                    model_dirs.append(item)
            
            structure_info[category_name] = {
                'model_count': model_count,
                'path': str(category_dir),
                'model_dirs': model_dirs[:5]  # Store first 5 for sampling
            }
            
            total_models += model_count
    
    print(f"\n📊 ShapeNet Structure Analysis:")
    print(f"  📂 Categories found: {len(structure_info)}")
    print(f"  🎯 Total models: {total_models}")
    print("-" * 60)
    
    for category, info in structure_info.items():
        print(f"  {category}: {info['model_count']} models")
    
    return structure_info, total_models

def sample_model_files(structure_info, sample_size=3):
    """Sample some model files to understand the data format"""
    print(f"\n🔬 Sampling model files (up to {sample_size} per category)...")
    
    file_types = {}
    
    for category, info in structure_info.items():
        if info['model_dirs']:
            print(f"\n📂 Category: {category}")
            
            for i, model_dir in enumerate(info['model_dirs'][:sample_size]):
                print(f"  Model {i+1}: {model_dir.name}")
                
                # List files in this model directory
                for file_path in model_dir.iterdir():
                    if file_path.is_file():
                        ext = file_path.suffix.lower()
                        if ext not in file_types:
                            file_types[ext] = 0
                        file_types[ext] += 1
                        print(f"    📄 {file_path.name}")
    
    print(f"\n📋 File type summary:")
    for ext, count in sorted(file_types.items()):
        print(f"  {ext}: {count} files")
    
    return file_types

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def cleanup_zip_files(zip_files, keep_zips=False):
    """Clean up downloaded zip files after extraction"""
    if keep_zips:
        print("\n💾 Keeping zip files as requested")
        return
    
    print(f"\n🧹 Cleaning up {len(zip_files)} zip files...")
    
    removed_count = 0
    for zip_path in zip_files:
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
                removed_count += 1
        except Exception as e:
            print(f"❌ Error removing {zip_path}: {e}")
    
    print(f"✓ Removed {removed_count} zip files")

def save_shapenet_manifest(structure_info, total_models, extract_dir):
    """Save a manifest of the ShapeNet data"""
    manifest = {
        "total_categories": len(structure_info),
        "total_models": total_models,
        "extract_directory": extract_dir,
        "categories": {}
    }
    
    for category, info in structure_info.items():
        manifest["categories"][category] = {
            "model_count": info["model_count"],
            "path": info["path"]
        }
    
    manifest_path = os.path.join(extract_dir, "shapenet_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"📄 Manifest saved to: {manifest_path}")
    return manifest_path

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def main():
    """Main function to run the ShapeNet loading process"""
    
    print("=" * 60)
    print("🎯 SHAPENET CORE LOADER")
    print("=" * 60)
    
    # Step 1: Setup and authentication
    client = setup_environment()
    if not client:
        return
    
    # Step 2: Connect to bucket
    bucket = connect_to_bucket(client, BUCKET_NAME)
    if not bucket:
        return
    
    # Step 3: Discover ShapeNet zip files
    zip_files = discover_shapenet_zips(bucket)
    if not zip_files:
        print("❌ No zip files found in the bucket!")
        return
    
    # Step 4: Show categorization
    categorized, uncategorized = categorize_zip_files(zip_files)
    
    # Step 4.5: Filter categories if specified
    if CATEGORIES_TO_EXTRACT is not None:
        zip_files = filter_zip_files_by_category(zip_files, CATEGORIES_TO_EXTRACT)
        if not zip_files:
            print("❌ No files match the specified categories!")
            return
    
    # Step 5: Confirm download
    total_size_gb = sum(zf['size_mb'] for zf in zip_files) / 1024
    print(f"\n❓ Ready to download {len(zip_files)} files ({total_size_gb:.2f} GB)?")
    print(f"   This will use approximately {total_size_gb * 2:.1f} GB of disk space (zip + extracted)")
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Download cancelled")
        return
    
    # Step 6: Download all zip files
    downloaded_files, failed_downloads = download_all_zips(bucket, zip_files, LOCAL_DATA_DIR)
    
    if not downloaded_files:
        print("❌ No files were downloaded successfully!")
        return
    
    # Step 7: Extract all zip files
    print(f"\n🔄 Starting extraction phase...")
    keep_zips = input("Keep zip files after extraction? (y/n): ").strip().lower() == 'y'
    
    extracted_dirs, failed_extractions, total_files = extract_all_zips(
        downloaded_files, EXTRACT_DIR, preserve_structure=True
    )
    
    # Step 8: Analyze structure
    structure_info, total_models = analyze_shapenet_structure(EXTRACT_DIR)
    
    # Step 9: Sample files
    if structure_info:
        sample_files = input("Sample model files to understand structure? (y/n): ").strip().lower() == 'y'
        if sample_files:
            file_types = sample_model_files(structure_info)
    
    # Step 10: Save manifest
    manifest_path = save_shapenet_manifest(structure_info, total_models, EXTRACT_DIR)
    
    # Step 11: Cleanup
    if not keep_zips:
        cleanup_zip_files(downloaded_files, keep_zips=False)
    
    # Final summary
    print(f"\n🎉 SHAPENET CORE LOADING COMPLETE!")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"  • Downloaded: {len(downloaded_files)} zip files")
    print(f"  • Extracted: {len(extracted_dirs)} archives")
    print(f"  • Total files: {total_files}")
    print(f"  • Categories: {len(structure_info)}")
    print(f"  • Total models: {total_models}")
    print(f"  • Data location: {EXTRACT_DIR}")
    print(f"  • Manifest: {manifest_path}")
    
    if failed_downloads:
        print(f"  ⚠️  Failed downloads: {len(failed_downloads)}")
    if failed_extractions:
        print(f"  ⚠️  Failed extractions: {len(failed_extractions)}")

if __name__ == "__main__":
    print("💡 Before running, make sure to update:")
    print(f"   • PROJECT_ID: {PROJECT_ID}")
    print(f"   • BUCKET_NAME: {BUCKET_NAME}")
    print("\n▶️  Starting ShapeNet Core loader...")
    
    main()