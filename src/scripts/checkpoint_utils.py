# scripts/checkpoint_utils.py
import os
import warnings

def fetch_checkpoint(bucket_name: str, blob_name: str, dest_path: str) -> str:
    """
    Download the checkpoint from GCS if it’s not already on disk.
    Returns the local path (whether newly-downloaded or already-present).
    """
    if os.path.exists(dest_path):
        return dest_path

    try:
        from google.cloud import storage
    except ImportError:
        warnings.warn("google-cloud-storage not installed; skipping checkpoint download")
        return dest_path

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(blob_name)
        print(f"⇣ Downloading gs://{bucket_name}/{blob_name} → {dest_path}")
        blob.download_to_filename(dest_path)
    except Exception as e:
        warnings.warn(f"Failed to download checkpoint ({e}); assuming it's already present")
    return dest_path
