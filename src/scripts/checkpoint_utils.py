# scripts/checkpoint_utils.py
import os
import warnings
import torch
import collections
from scripts.joint_regressor           import JointRegressor

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

def load_pretrained_backbone(model: JointRegressor, ckpt_path: str):
    """
    Load tokenizer, positional encoding, encoder, and pool weights from a checkpoint.
    Verifies that all expected keys were loaded and prints mismatches.
    """
    print(f"📂 Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    prefixes = ("tokenizer.", "positional_encoding.", "encoder.", "pool.")
    model_state = model.state_dict()
    loadable = {}
    expected_keys = []

    for k, v in state.items():
        if k.startswith(prefixes):
            expected_keys.append(k)
            if k in model_state and model_state[k].shape == v.shape:
                loadable[k] = v
            else:
                print(f"⚠️ Skipping {k}: shape mismatch "
                      f"(ckpt={tuple(v.shape)}, model={tuple(model_state.get(k, torch.tensor([])).shape)})")

    # Load the filtered state dict
    missing, unexpected = model.load_state_dict(loadable, strict=False)

    print(f"\n✅ Loaded {len(loadable)} tensors.")
    print(f"❌ Missing keys after load: {len(missing)}")
    print(f"⚠️ Unexpected keys: {len(unexpected)}")

    # Extra verification: check if all expected keys were loaded
    missing_from_ckpt = [k for k in expected_keys if k not in loadable]
    if missing_from_ckpt:
        print(f"\n🚨 WARNING: {len(missing_from_ckpt)} expected keys not loaded due to mismatches:")
        for k in missing_from_ckpt:
            print(f"   - {k}")
    else:
        print("\n✅ All expected keys loaded successfully.")


