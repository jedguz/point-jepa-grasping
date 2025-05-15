import os, sys
# Insert voxel-mae submodule into path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), 'ext/voxel-mae')))

import torch
from torch.utils.data import DataLoader
from voxel_mae.models import VoxelMAE
from src.data.tsdf_dataset import TSDFVoxelDataset

# Detect device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 1) Load the pretrained checkpoint
ckpt_path = 'ext/voxel-mae/pretrained/voxelmae_nuscenes.pth'
model = VoxelMAE(
    in_channels=1,
    embed_dims=384,
    patch_size=(2,4,4),
    mask_ratio=0.0,   # no masking when extracting features
)
state = torch.load(ckpt_path, map_location=device)
# The checkpoint key may be 'state_dict' or directly weights
if 'state_dict' in state:
    model.load_state_dict(state['state_dict'], strict=False)
else:
    model.load_state_dict(state, strict=False)
model = model.to(device)
model.eval()

# 2) Prepare dataset & loader
dataset = TSDFVoxelDataset(root_dir='data/tsdf_npy')
loader = DataLoader(dataset, batch_size=1, num_workers=0)

# 3) Extract embeddings
emb_list = []
with torch.no_grad():
    for batch in loader:
        x = batch['points'].to(device)
        feats = model.encoder(x)   # shape [B, embed_dims]
        emb_list.append(feats.cpu())

embeddings = torch.cat(emb_list, dim=0)
torch.save(embeddings, 'tsdf_embeddings.pt')
print(f"Saved {embeddings.shape[0]} embeddings to tsdf_embeddings.pt")