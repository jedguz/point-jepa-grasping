#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np

# --- make sure Python can find src/
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src  = os.path.join(root, "src")
if src not in sys.path:
    sys.path.insert(0, src)

from pipeline.point2vec_3djepa import Point2Vec3DJEPA

def load_pc(path):
    data = np.load(path)
    pc = data.get("points", data.get("pc_2048", data.get("pc")))
    if pc is None:
        raise KeyError(f"No points array found in {path}")
    return torch.from_numpy(pc).unsqueeze(0).float()  # (1, N, 3)

if __name__ == "__main__":
    # --- CONFIG ---
    pc_path        = "data/pointclouds/02691156_10aa040f470500c6a66ef8df4909ded9_2048.npz"
    tokenizer_ckpt = "checkpoints/point2vec_tokenizer_only.pt"
    tokenizer_cfg  = dict(
        num_groups=256,
        group_size=32,
        group_radius=None,
        token_dim=384,
    )
    encoder_ckpt   = None
    encoder_cfg    = dict(
        # must be 2 * tokenizer_cfg["token_dim"]
        input_feat_dim=384 * 2,   # original JEPA3D expects 2 streams
        embed_dim=768,
        rgb_proj_dim=64,
        num_rgb_harmonic_functions=16,
        ptv3_args={"in_channels": 768},
        voxel_size=0.05,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- BUILD PIPELINE ---
    pipe = Point2Vec3DJEPA(
        tokenizer_ckpt=tokenizer_ckpt,
        tokenizer_cfg=tokenizer_cfg,
        encoder_cfg=encoder_cfg,
        encoder_ckpt=encoder_ckpt,
        device=device,
    )

    # **IMPORTANT**: switch EVERYTHING to eval so BatchNorm won't complain
    
    pipe.to(device)
    pipe.eval()

    import torch.nn as nn

    print("=== BatchNorm modules and their .training flags ===")
    for name, m in pipe.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            print(f"{name:40s}│ training={m.training}  track_running_stats={m.track_running_stats}")
    print("====================================================")
    
    # --- RUN FORWARD ---
    pc = load_pc(pc_path).to(device)
    with torch.no_grad():
        out = pipe(pc)

    feats   = out.get("jepa_feats", out.get("features"))
    centers = out["centers"]

    print("JEPA features shape :", feats.shape)   # expect [1, G, 768]
    print("Patch centers shape :", centers.shape) # expect [1, G,   3]
