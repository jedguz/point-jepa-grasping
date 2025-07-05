# scripts/checkpoint_utils.py
import os
import warnings
import torch
import torch.nn as nn
import collections
from scripts.joint_regressor import JointRegressor
from utils.checkpoint import extract_model_checkpoint 

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

def load_full_checkpoint(model: JointRegressor, path: str) -> None:
    print(f"Loading pretrained checkpoint from '{path}'.")

    checkpoint = extract_model_checkpoint(path)

    for k in list(checkpoint.keys()):
        if k.startswith("cls_head."):
            del checkpoint[k]
        elif k.startswith("head."):
            del checkpoint[k]
        elif k.startswith("predictor."):
            del checkpoint[k]
    
    # missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)  # type: ignore
    # print(f"Missing keys: {missing_keys}")
    # print(f"Unexpected keys: {unexpected_keys}")
    
    # fix NaNs in batchnorm, this has been observed in some checkpoints... not sure why
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            if torch.any(torch.isnan(m.running_mean)):  # type: ignore
                print(f"Warning: NaNs in running_mean of {name}. Setting to zeros.")
                m.running_mean = torch.zeros_like(m.running_mean)  # type: ignore
            if torch.any(torch.isnan(m.running_var)):  # type: ignore
                print(f"Warning: NaNs in running_var of {name}. Setting to ones.")
                m.running_var = torch.ones_like(m.running_var)  # type: ignore



