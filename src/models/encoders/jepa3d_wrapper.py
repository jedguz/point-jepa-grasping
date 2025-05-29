# my_project/models/jepa3d_wrapper.py

import torch
import torch.nn as nn
from ext.jepa3d.models.encoder_3djepa import Encoder3DJEPA

class JEPA3DEncoderWrapper(nn.Module):
    """
    Thin wrapper around Locate-3D's Encoder3DJEPA that:
      1) loads pretrained weights
      2) freezes the encoder
      3) pools per-point outputs to a single global vector of size [B, embed_dim]
    """
    def __init__(
        self,
        input_feat_dim: int = 512,
        embed_dim:       int = 768,
        rgb_proj_dim:   int = 128,
        ptv3_args:      dict = {},
        voxel_size:     float = 0.05,
        checkpoint_path: str = None,
    ):
        super().__init__()
        # 1) instantiate with the same args the paper uses
        self.encoder = Encoder3DJEPA(
            input_feat_dim=input_feat_dim,
            embed_dim=embed_dim,
            rgb_proj_dim=rgb_proj_dim,
            ptv3_args=ptv3_args,
            voxel_size=voxel_size,
        )

        # 2) load their checkpoint (adjust key-names if needed)
        if checkpoint_path is not None:
            self.encoder.load_weights(checkpoint_path)

        # 3) freeze everything
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, featurized_scene: dict):
        """
        featurized_scene must contain:
           - 'features_clip': Tensor[N, c_clip]
           - 'features_dino': Tensor[N, c_dino]
           - 'rgb'          : Tensor[N, 3]           (values in [0,1])
           - 'points'       : Tensor[N, 3]           (voxel-center coords)
        Returns:
           - global_feat: Tensor[B=1, embed_dim]
        """
        out = self.encoder(featurized_scene)
        # out['features'] is shape [N, embed_dim]
        pts_feat = out['features']
        # simple mean pooling to get a single vector per batch
        global_feat = pts_feat.mean(dim=0, keepdim=True)  # [1, embed_dim]
        return global_feat
