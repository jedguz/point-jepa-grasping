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

def load_full_checkpoint(model, ckpt_path: str):
    """
    Loads the full checkpoint into the model with shape validation.
    Skips keys with mismatched shapes and logs all important stats.
    """
    print(f"📂 Loading full checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    model_state = model.state_dict()
    loadable = {}
    skipped = []

    for k, v in state.items():
        if k in model_state and model_state[k].shape == v.shape:
            loadable[k] = v
        else:
            model_shape = model_state.get(k, torch.tensor([])).shape
            skipped.append((k, v.shape, model_shape))

    # Load valid weights only
    missing, unexpected = model.load_state_dict(loadable, strict=False)

    print(f"\n✅ Loaded {len(loadable)} tensors into model.")
    print(f"❌ Missing keys in model: {len(missing)}")
    print(f"⚠️ Unexpected keys in checkpoint: {len(unexpected)}")

    if skipped:
        print(f"\n🚫 Skipped {len(skipped)} keys due to shape mismatch:")
        for k, v_shape, m_shape in skipped:
            print(f"   - {k}: ckpt={tuple(v_shape)}, model={tuple(m_shape)}")

    if not missing and not skipped:
        print("\n🎉 All checkpoint parameters successfully loaded!")

