import torch
from models.encoders.jepa3d_backbone import Encoder

encoder = Encoder(pretrained=False)
encoder.eval()

# Dummy pointcloud: (B=1, N=1024, C=3)
pc = torch.randn(1, 1024, 3)
with torch.no_grad():
    feats = encoder(npc)
print('Embedding shape:', feats.shape)  # should be [1, D]