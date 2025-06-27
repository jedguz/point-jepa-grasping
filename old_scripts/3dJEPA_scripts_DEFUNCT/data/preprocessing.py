#!/usr/bin/env python3
import os
import argparse
import glob
import numpy as np
import torch

# adjust this to wherever your PointNet2Embed class lives:
from src.models.encoders.pointnet2 import PointNet2Embed

def load_npz_pointcloud(path):
    data = np.load(path)
    pc = data[data.files[0]]        # shape (N,3) or (N,6)
    # ensure (3, N) float32
    pc = torch.from_numpy(pc.astype(np.float32)).transpose(0,1).unsqueeze(0)
    return pc  # shape (1, C, N)

def main(args):
    # instantiate embedder
    model = PointNet2Embed(pretrained_path=args.checkpoint)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # find all .npz pointclouds
    pattern = os.path.join(args.input_dir, "*", "pc_2048.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No files found at {pattern}")

    for pc_path in files:
        # output path in same folder
        out_path = os.path.join(
            os.path.dirname(pc_path),
            args.output_key
        )
        if os.path.exists(out_path):
            print(f"Skipping (exists): {out_path}")
            continue

        # load, featurize, save
        pc = load_npz_pointcloud(pc_path).to(device)        # (1,3,2048)
        with torch.no_grad():
            emb = model(pc)                                 # (1,1024)
        emb = emb.squeeze(0).cpu().numpy()                 # (1024,)

        np.save(out_path, emb)
        print(f"Saved embedding: {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Precompute PointNet2 embeddings for pointclouds"
    )
    p.add_argument("--input-dir", required=True,
                   help="root folder containing subfolders with pc_2048.npz")
    p.add_argument("--checkpoint", required=True,
                   help="path to pretrained pointnet2.pth")
    p.add_argument("--output-key", default="emb_1024.npy",
                   help="filename for saved embeddings")
    args = p.parse_args()
    main(args)
