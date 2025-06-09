#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np

# Add src/ to PYTHONPATH
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src  = os.path.join(root, "src")
if src not in sys.path:
    sys.path.insert(0, src)

from pipeline.point2vec_3djepa import Point2Vec3DJEPA

def load_pc(path):
    data = np.load(path)
    pc = data.get("points", data.get("pc_2048", data.get("pc")))
    return torch.from_numpy(pc).unsqueeze(0).float()  # (1, N, 3)

if __name__ == "__main__":
    # --- CONFIG ---
    pc_path = "data/pointclouds/02691156_10aa040f470500c6a66ef8df4909ded9_2048.npz"
    tokenizer_ckpt = "checkpoints/point2vec_tokenizer_only.pt"
    tokenizer_cfg = dict(num_groups=256, group_size=32, group_radius=None, token_dim=384)
    encoder_cfg   = dict(embed_dim=768, rgb_proj_dim=64, num_rgb_harmonic_functions=16, ptv3_args={}, voxel_size=0.05)

    # --- RUN ---
    pc = load_pc(pc_path).to("cpu")  # or "cuda"
    pipe = Point2Vec3DJEPA(tokenizer_ckpt, tokenizer_cfg, encoder_ckpt=None, encoder_cfg=encoder_cfg, device="cpu")
    features, points = pipe.forward(pc)["features"], pipe.forward(pc)["points"]

    print("Output token embeddings:", features.shape)  # (1, G, embed_dim)
    print("Output patch centers  :", points.shape)    # (1, G,   3   )
