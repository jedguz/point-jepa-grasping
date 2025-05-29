# src/models/pn2jepa_connector.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# import the SA module from your vendored PyTorch PN++ repo
from ext.pointnet2.pointnet2_utils import PointNetSetAbstraction 

from models.encoders.jepa3d_wrapper import JEPA3DEncoderWrapper

class PN2JEPAConnector(nn.Module):
    def __init__(self, *, jepa_ckpt:str, jepa_embed_dim:int=768):
        super().__init__()
        # 1) one SA layer: 512 centroids, mlp->[64,64,128]
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=3, mlp=[64,64,128],
            group_all=False
        )
        # freeze SA1 if you like
        for p in self.sa1.parameters(): p.requires_grad = False

        # 2) JEPA3D wrapper expects input_feat_dim=128
        self.jepa = JEPA3DEncoderWrapper(
            input_feat_dim=128,
            embed_dim=jepa_embed_dim,
            rgb_proj_dim=0,          # no RGB/CLIP features
            ptv3_args={},
            voxel_size=0.05,
            checkpoint_path=jepa_ckpt,
        )

    def forward(self, xyz:torch.Tensor):
        """
        xyz: [N,3] or [1,N,3] float32
        returns: [1, jepa_embed_dim] global JEPA3D feature
        """
        if xyz.dim()==2:
            xyz = xyz.unsqueeze(0)      # [1,N,3]
        B,N,_ = xyz.shape

        # PN++ SA1 expects pts=[B,3,N], normals=None
        pts = xyz.permute(0,2,1)
        l1_xyz, l1_feats = self.sa1(pts, None)
        # l1_xyz: [B,3,512], l1_feats: [B,128,512]

        # reshape into flat 512 tokens
        centroids = l1_xyz.permute(0,2,1).reshape(-1,3)     # [512,3]
        feats     = l1_feats.permute(0,2,1).reshape(-1,128) # [512,128]

        # build JEPA3D input dict
        featurized = {
            "points":        centroids,
            "rgb":           torch.zeros_like(centroids),
            "features_dino": feats,
            "features_clip": torch.zeros_like(feats),
        }

        # run JEPA → returns [1, jepa_embed_dim]
        return self.jepa(featurized)
