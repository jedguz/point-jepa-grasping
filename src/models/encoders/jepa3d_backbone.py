import torch
import torch.nn as nn

# Use the Encoder3DJEPA class directly from the Locate-3D submodule
from ext.jepa3d.models.encoder_3djepa import Encoder3DJEPA

class Encoder(nn.Module):
    """
    3D-JEPA backbone wrapper.  forward(x) -> features [B, D]
    """
    def __init__(self, pretrained=True, checkpoint_path=None):
        super().__init__()
        # instantiate the raw JEPA3D encoder
        self.model = Encoder3DJEPA()
        if pretrained and checkpoint_path:
            ck = torch.load(checkpoint_path, map_location='cpu')
            sd = ck.get('state_dict', ck)
            self.model.load_state_dict(sd, strict=False)
        # freeze encoder parameters
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        # x: [B, C, D, H, W] TSDF grid or [B, N, 3] pointcloud
        return self.model.encode(x)