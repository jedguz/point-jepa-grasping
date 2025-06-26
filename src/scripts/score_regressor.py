# scripts/grasp_regressor.py
"""
GraspRegressor – pointJEPA-style network that regresses a scalar grasp score
from (point cloud, 19-D grasp vector).

Changes vs. previous version
----------------------------
1. All trainable sub-modules are now passed to the optimiser.
2. Head outputs shape (B, 1); losses call .squeeze(-1).
3. Added a cosine LR scheduler (easy to remove if not wanted).
4. Tiny logging tweaks and a safer return value in validation_step.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from modules.tokenizer import PointcloudTokenizer
from modules.transformer import TransformerEncoder
from scripts.pooling import get_pooling
from torch.utils.data import DataLoader, TensorDataset


class GraspRegressor(pl.LightningModule):
    # ------------------------------------------------------------------- init
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
        pooling_type: str,
        pooling_heads: int,
        head_hidden_dims: list[int],
        head_output_dim: int = 1,
        grasp_dim: int = 19,
        lr_backbone: float = 3e-4,
        lr_head: float = 3e-3,
        weight_decay: float = 1e-2,
    ):
        """
        Parameters marked with * are forwarded directly from Hydra configs.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["head_hidden_dims"])

        # ── tokenizer
        self.tokenizer = PointcloudTokenizer(
            num_groups=tokenizer_groups,
            group_size=tokenizer_group_size,
            group_radius=tokenizer_radius,
            token_dim=encoder_dim,
        )

        # ── positional encoding on group centroids
        self.positional_encoding = nn.Sequential(
            nn.Linear(3, encoder_dim),
            nn.GELU(),
            nn.Linear(encoder_dim, encoder_dim),
        )

        # ── transformer backbone
        dpr = torch.linspace(0, encoder_drop_path_rate, encoder_depth).tolist()
        self.encoder = TransformerEncoder(
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
            drop_rate=encoder_dropout,
            attn_drop_rate=encoder_attn_dropout,
            drop_path_rate=dpr,
            add_pos_at_every_layer=True,
        )

        # ── pooling
        self.pool = get_pooling(pooling_type, dim=encoder_dim, num_heads=pooling_heads)

        # ── MLP head
        in_dim = encoder_dim + grasp_dim
        dims = [in_dim] + head_hidden_dims + [head_output_dim]
        mlp: list[nn.Module] = []
        for i in range(len(dims) - 2):
            mlp.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        mlp.append(nn.Linear(dims[-2], dims[-1]))
        self.head = nn.Sequential(*mlp)


    # ---------------------------------------------------------------- forward
    @torch.cuda.amp.autocast(enabled=True)
    def forward(self, points: torch.Tensor, grasp_vec: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        points : (B, N, 3)
        grasp_vec : (B, 19)

        Returns
        -------
        (B, 1) tensor of predicted scores
        """
        tokens, centers = self.tokenizer(points)          # (B, L, D), (B, L, 3)
        pos = self.positional_encoding(centers)           # (B, L, D)

        feats = self.encoder(tokens, pos).last_hidden_state      # (B, L, D)
        obj_emb = self.pool(feats)                        # (B, D)

        combined = torch.cat([obj_emb, grasp_vec], dim=-1)       # (B, D+19)
        return self.head(combined)                        # (B, 1)
    
    # ------------------------------------------------------------------- training
    def training_step(self, batch, batch_idx):
        points, grasp_vec, score = batch
        pred = self(points, grasp_vec).squeeze(-1)
        loss = F.mse_loss(pred, score)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        points, grasp_vec, score = batch
        pred = self(points, grasp_vec).squeeze(-1)
        loss = F.mse_loss(pred, score)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        # freeze backbone if lr_backbone == 0
        backbone_params, head_params = [], []
        if self.hparams.lr_backbone > 0:
            backbone_params = [
                {"params": self.tokenizer.parameters(),          "lr": self.hparams.lr_backbone},
                {"params": self.positional_encoding.parameters(),"lr": self.hparams.lr_backbone},
                {"params": self.encoder.parameters(),            "lr": self.hparams.lr_backbone},
            ]
        head_params = {"params": self.head.parameters(), "lr": self.hparams.lr_head}

        optimizer = torch.optim.AdamW(backbone_params + [head_params],
                                    weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}