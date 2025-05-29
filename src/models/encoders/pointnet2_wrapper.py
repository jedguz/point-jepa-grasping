# src/models/pointnet2_wrapper.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your PyTorch PointNet++ model
from pointnet2_pytorch.models.pointnet2_cls_ssg import Pointnet2SSG

class PointNet2Wrapper(nn.Module):
    """
    Wraps the imported PointNet++ SSG classifier to produce per-point embeddings.
    We'll take the final 'global' feature and replicate it across each input point.
    """
    def __init__(self, num_class=40, input_dim=3, normal_channel=False, pretrained_ckpt=None):
        super().__init__()
        # 1) Instantiate the PN++ backbone
        self.backbone = Pointnet2SSG(
            in_channels=(input_dim + (3 if normal_channel else 0)),
            num_classes=num_class
        )
        if pretrained_ckpt:
            self.backbone.load_state_dict(torch.load(pretrained_ckpt))
        self.backbone.eval()  # freeze for POC
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, xyz: torch.Tensor):
        """
        xyz: [B, N, 3] (or [N,3], we'll unsqueeze)
        Returns:
          per_pt_feats: [B, N, C]  where C is the 1024‐dim global feature
        """
        if xyz.dim() == 2:
            xyz = xyz.unsqueeze(0)        # -> [1,N,3]
        B, N, _ = xyz.shape
        # PN++ expects [B, C, N]
        pts = xyz.permute(0, 2, 1)       # [B,3,N]
        logits, global_feat = self.backbone(pts)  
        # global_feat: [B, 1024] final layer before classifier
        # replicate per point:
        per_pt = global_feat.unsqueeze(2).repeat(1, 1, N)  # [B,1024,N]
        # back to [B,N,1024]
        per_pt = per_pt.permute(0, 2, 1)
        return per_pt  # this is your initial embedding per point
