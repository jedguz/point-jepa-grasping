import torch
import torch.nn as nn
from ext.point2vec.tokenizer import PointCloudTokenizer
from ext.jepa3d.models.encoder_3djepa import Encoder3DJEPA

class Point2Vec3DJEPA(nn.Module):
    """
    Pipeline combining Point2Vec tokenizer + optional 3D-JEPA encoder.
    If encoder_cfg is provided, the pipeline will run both stages;
    otherwise it returns just tokens and centers.
    """
    def __init__(
        self,
        tokenizer_ckpt: str,
        tokenizer_cfg: dict,
        encoder_cfg: dict = None,
        encoder_ckpt: str = None,
        device: str = "cuda",
    ):
        super().__init__()
        # 1) build tokenizer
        self.tokenizer = PointCloudTokenizer(**tokenizer_cfg)
        self.tokenizer.load_state_dict(torch.load(tokenizer_ckpt, map_location="cpu"))
        self.tokenizer.eval()

        # 2) optionally build JEPA3D encoder
        if encoder_cfg is not None:
            self.encoder = Encoder3DJEPA(**encoder_cfg)
            if encoder_ckpt:
                self.encoder.load_weights(encoder_ckpt)
            self.encoder.eval()
        else:
            self.encoder = None

        self.device = device
        self.to(device)

    @torch.no_grad()
    def forward(self, pointcloud: torch.Tensor, rgb: torch.Tensor = None):
        """
        pointcloud: (B, N, 3)
        rgb       : (B, N, 3) optional, passed only if encoder is used

        Returns:
          If encoder is None:
            {"features": (B, G, token_dim), "points": (B, G, 3)}
          If encoder is provided:
            {"jepa_feats": (B, G, embed_dim), "centers": (B, G, 3)}
        """
        B, N, _ = pointcloud.shape
        pointcloud = pointcloud.to(self.device)

        # Tokenize
        tokens, centers = self.tokenizer(pointcloud)
        # tokens: (B, G, C), centers: (B, G, 3)

        if self.encoder is None:
            return {"features": tokens, "points": centers}

        # Build featurized dict for JEPA3D
        # dummy second stream for compatibility
        dummy = torch.zeros_like(tokens)
        rgb_input = rgb.to(self.device) if rgb is not None else torch.zeros_like(centers)

        if B == 1:
            feats = tokens.squeeze(0)
            d_feats = dummy.squeeze(0)
            ctrs  = centers.squeeze(0)
            rgb0  = rgb_input.squeeze(0)

            featurized = {
                "features_clip": feats,
                "features_dino": d_feats,
                "points": ctrs,
                "rgb": rgb0,
            }
            out = self.encoder(featurized)
            # out["features"]: (G, D)
            jepa_feats = out["features"].unsqueeze(0)
            return {"jepa_feats": jepa_feats, "centers": ctrs.unsqueeze(0)}
        else:
            # Batch>1 support requires modifying Encoder3DJEPA as well
            raise NotImplementedError("Batch size >1 not supported yet.")
