#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np

# 1) point to your src/ so “pipeline” can import
root   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src    = os.path.join(root, "src")
if src not in sys.path:
    sys.path.insert(0, src)

from pipeline.point2vec_3djepa import Point2Vec3DJEPA

def load_pc(path):
    data = np.load(path)
    pc = data.get("points", data.get("pc_2048", data.get("pc")))
    return torch.from_numpy(pc[None, ...]).float()  # (1, N, 3)

if __name__ == "__main__":
    # --- CONFIG ---
    pc_path = "data/pointclouds/02691156_10aa040f470500c6a66ef8df4909ded9_2048.npz"
    tokenizer_ckpt = "checkpoints/point2vec_tokenizer_only.pt"
    tokenizer_cfg = dict(
        num_groups=256,
        group_size=32,
        group_radius=None,
        token_dim=384,
    )
    # encoder_cfg for a random 3D-JEPA (no ckpt)
    encoder_cfg = dict(
        input_feat_dim=384*2,   # 384 clip + 384 “dummy” dino
        embed_dim=768,
        rgb_proj_dim=64,
        num_rgb_harmonic_functions=16,
        ptv3_args={}, 
        voxel_size=0.05,
    )

    # --- INSTANTIATE FULL PIPELINE ---
    pipe = Point2Vec3DJEPA(
        tokenizer_ckpt=tokenizer_ckpt,
        tokenizer_cfg=tokenizer_cfg,
        encoder_cfg=encoder_cfg,   # <— include this to build JEPA
        encoder_ckpt=None,         # <— start with random weights
        device="cpu",
    )

    # --- RUN ON A SINGLE CLOUD ---
    pc = load_pc(pc_path)
    out = pipe(pc)

    # --- PRINT SHAPES ---
    print("Token embeddings  :", out["features"].shape)   # (1, G, 384)
    print("Patch centers     :", out["points"].shape)     # (1, G,   3)
    print("JEPA output feats :", out["jepa_feats"].shape)  # (1, G, 768)
    print("JEPA output ctrs  :", out["centers"].shape)     # (1, G,   3)
