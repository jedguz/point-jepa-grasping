"""
JointRegressor – predicts the 12-D finger configuration given:
  • point cloud of the object
  • 7-D hand pose (translation + quaternion)
"""

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F, pytorch_lightning as pl
from modules.tokenizer   import PointcloudTokenizer
from modules.transformer import TransformerEncoder
from modules.transformer import TransformerPredictor
from scripts.pooling     import get_pooling


class JointRegressor(pl.LightningModule):
    # --------------------------------------------------------------------- init
    def __init__(
        self,
        *,
        num_points: int,
        tokenizer_groups: int,
        tokenizer_group_size: int,
        tokenizer_radius: float,
        encoder_dim: int,
        encoder_depth: int,
        encoder_heads: int,
        encoder_dropout: float,
        encoder_attn_dropout: float,
        encoder_drop_path_rate: float,
        encoder_mlp_ratio: float,
        pooling_type: str,
        pooling_heads: int,
        head_hidden_dims: list[int],
        pose_dim: int = 7,               # always 7
        lr_backbone: float = 3e-4,
        lr_head: float = 3e-3,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["head_hidden_dims"])

        # tokenizer
        self.tokenizer = PointcloudTokenizer(
            num_groups  = tokenizer_groups,
            group_size  = tokenizer_group_size,
            group_radius= tokenizer_radius,
            token_dim   = encoder_dim,
        )

        # positional encoding on centroids
        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, encoder_dim)  # 123 hidden dim fixed in prerained backbone
        )

        # transformer backbone
        dpr = torch.linspace(0, encoder_drop_path_rate, encoder_depth).tolist()
        self.encoder = TransformerEncoder(
            embed_dim      = encoder_dim,
            depth          = encoder_depth,
            num_heads      = encoder_heads,
            mlp_ratio      = encoder_mlp_ratio,
            qkv_bias       = True,
            drop_rate      = encoder_dropout,
            attn_drop_rate = encoder_attn_dropout,
            drop_path_rate = dpr,
            add_pos_at_every_layer=True,
        )

        # pooling
        self.pool = get_pooling(pooling_type, dim=encoder_dim, num_heads=pooling_heads)

        # MLP head → 12 numbers
        in_dim = encoder_dim + pose_dim
        dims   = [in_dim] + head_hidden_dims + [12]
        mlp: list[nn.Module] = []
        for i in range(len(dims) - 2):
            mlp.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        mlp.append(nn.Linear(dims[-2], dims[-1]))
        self.head = nn.Sequential(*mlp)

    # ------------------------------------------------------------------ forward
    @torch.cuda.amp.autocast(enabled=True)
    def forward(self, points: torch.Tensor, pose_vec: torch.Tensor) -> torch.Tensor:
        """
        points   : (B, N, 3)
        pose_vec : (B, 7)
        returns  : (B, 12) predicted joint angles
        """
        tokens, centers = self.tokenizer(points)          # (B,L,D), (B,L,3)
        pos = self.positional_encoding(centers)           # (B,L,D)
        feats = self.encoder(tokens, pos).last_hidden_state
        obj_emb = self.pool(feats)                        # (B,D)
        fused = torch.cat([obj_emb, pose_vec], dim=-1)     # (B,D+7)
        return self.head(fused)                           # (B,12)

    # ------------------------------------------------------------- train / val
    def _step(self, batch, stage: str):
        pred = self(batch["points"], batch["pose"])       # (B,12)
        loss = F.mse_loss(pred, batch["joints"])
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step  (self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): self._step(batch, "val")

    # ----------------------------------------------------------------- optimizer
    def configure_optimizers(self):
        # freeze backbone if lr_backbone == 0
        bb_params = []
        if self.hparams.lr_backbone > 0:
            bb_params = [
                {"params": self.tokenizer.parameters(),           "lr": self.hparams.lr_backbone},
                {"params": self.positional_encoding.parameters(), "lr": self.hparams.lr_backbone},
                {"params": self.encoder.parameters(),             "lr": self.hparams.lr_backbone}
            ]
        head_params = {"params": self.head.parameters(), "lr": self.hparams.lr_head}

        optim = torch.optim.AdamW(bb_params + [head_params], weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.trainer.max_epochs)
        return {"optimizer": optim, "lr_scheduler": sched}
