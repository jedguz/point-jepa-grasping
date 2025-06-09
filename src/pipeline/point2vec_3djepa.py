import torch
import torch.nn as nn
from ext.point2vec.tokenizer import PointCloudTokenizer
from ext.jepa3d.models.encoder_3djepa import Encoder3DJEPA

class Point2Vec3DJEPA(nn.Module):
    def __init__(
        self,
        tokenizer_ckpt: str,
        tokenizer_cfg: dict,
        encoder_cfg: dict = None,
        encoder_ckpt: str = None,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device

        # 1) Build & eval‐mode the tokenizer
        self.tokenizer = PointCloudTokenizer(**tokenizer_cfg).to(device)
        self.tokenizer.load_state_dict(torch.load(tokenizer_ckpt, map_location="cpu"))
        self.tokenizer.eval()

        token_dim = self.tokenizer.token_dim  # 384

        # 2) Build the original two‐stream JEPA3D encoder (unmodified), force it onto device & eval
        if encoder_cfg is not None:
            # tell it we have two 384-dim streams = 768 total
            encoder_cfg = {
                **encoder_cfg,
                "input_feat_dim": token_dim * 2,
                # ensure PointTransformerV3 inside expects embed_dim channels
                "ptv3_args": {
                    **encoder_cfg.get("ptv3_args", {}),
                    "in_channels": encoder_cfg["embed_dim"],
                },
            }
            self.encoder = Encoder3DJEPA(**encoder_cfg).to(device)
            if encoder_ckpt:
                self.encoder.load_weights(encoder_ckpt)
            # global eval on encoder & all submodules
            self.encoder.eval()
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()
        else:
            self.encoder = None

    @torch.no_grad()
    def forward(self, pointcloud: torch.Tensor, rgb: torch.Tensor = None):
        """
        pointcloud: (B, N, 3)
        rgb       : (B, N, 3) or None
        """
        pointcloud = pointcloud.to(self.device)
        B, N, _   = pointcloud.shape

        # 3) Tokenize
        tokens, centers = self.tokenizer(pointcloud)   # (B, G, 384), (B, G, 3)

        if self.encoder is None:
            return {"features": tokens, "points": centers}

        # 4) Make absolutely sure every BN1d in the encoder is still in eval
        #    (in case something upstream switched train() back on)
        self.encoder.eval()
        for m in self.encoder.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

        # 5) Duplicate your single stream into both CLIP + DINO slots
        feats     = tokens
        d_feats   = tokens.clone()
        rgb_input = rgb.to(self.device) if rgb is not None else torch.zeros_like(centers)

        # only batch=1 supported:
        feats   = feats.squeeze(0)     # (G,384)
        d_feats = d_feats.squeeze(0)   # (G,384)
        ctrs    = centers.squeeze(0)   # (G,  3)
        rgb0    = rgb_input.squeeze(0) # (G,  3)

        featurized = {
            "features_clip": feats,
            "features_dino": d_feats,
            "points":        ctrs,
            "rgb":           rgb0,
        }

        # 6) Forward through the unaltered JEPA3D encoder
        out = self.encoder(featurized)

        # 7) Re-pack batch dimension
        return {
            "jepa_feats": out["features"].unsqueeze(0),  # (1, G, embed_dim)
            "centers":    ctrs.unsqueeze(0),              # (1, G, 3)
        }
