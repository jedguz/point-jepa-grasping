# ──────────────────────────────────────────────────────────────
# scripts/joint_regressor_mdn.py   
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import os, math
import torch, torch.nn as nn, torch.nn.functional as F, pytorch_lightning as pl
import wandb

from modules.tokenizer   import PointcloudTokenizer
from modules.transformer import TransformerEncoder
from scripts.pooling     import get_pooling
from scripts.mdn         import MDNHead, mdn_nll

class JointRegressorMDN(pl.LightningModule):
    def __init__(self,
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
        mdn_mixtures: int = 10,
        pose_dim: int = 7,
        lr_backbone: float = 3e-4,
        lr_head: float = 1e-3,
        weight_decay: float = 1e-2,
        beta_entropy: float = 1e-3,      # ← encourage mixture balance
        var_min: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ─ backbone ─
        self.tokenizer = PointcloudTokenizer(
            num_groups   = tokenizer_groups,
            group_size   = tokenizer_group_size,
            group_radius = tokenizer_radius,
            token_dim    = encoder_dim,
        )
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
        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, encoder_dim)
        )
        self.pool = get_pooling(pooling_type, dim=encoder_dim, num_heads=pooling_heads)

        # ─ MDN head ─
        self.mdn = MDNHead(
            in_dim       = encoder_dim + pose_dim,
            out_dim      = 12,
            num_mixtures = mdn_mixtures,
            var_min      = var_min,
        )

    # -------------------------------------------------------- forward
    @torch.cuda.amp.autocast(enabled=True)
    def forward(self, points: torch.Tensor, pose_vec: torch.Tensor):
        tokens, centers = self.tokenizer(points)                 # (B,L,D)
        pos   = self.positional_encoding(centers)
        feats = self.encoder(tokens, pos).last_hidden_state
        obj_emb = self.pool(feats)                               # (B,D)
        fused   = torch.cat([obj_emb, pose_vec], dim=-1)
        return self.mdn(fused)                                   # log_pi, μ, σ

    # --------------------------------------------------- shared step
    def _step(self, batch, stage: str):
        log_pi, mu, sigma = self(batch["points"], batch["pose"])
        nll = mdn_nll(log_pi, mu, sigma, batch["joints"])

        # entropy regulariser
        pi = log_pi.exp()
        entropy = (-pi * log_pi).sum(1).mean()
        loss = nll - self.hparams.beta_entropy * entropy

        # logging
        self.log(f"{stage}_loss", loss,    prog_bar=True)
        self.log(f"{stage}_nll",  nll,     prog_bar=False)
        self.log(f"{stage}_Hpi",  entropy, prog_bar=False)

        # histogram once per epoch (first batch)
        if stage == "train" and self.global_step % self.trainer.num_training_batches == 0:
            self.logger.experiment.log(
                {"pi_hist": wandb.Histogram(pi.detach().cpu().numpy())},
                step=self.global_step,
            )
        return loss

    def training_step  (self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): self._step(batch, "val")

    # ------------------------------------------------ optim & sched
    def configure_optimizers(self):
        bb_params = []
        if self.hparams.lr_backbone > 0:
            bb_params = [
                {"params": self.tokenizer.parameters(),           "lr": self.hparams.lr_backbone},
                {"params": self.positional_encoding.parameters(), "lr": self.hparams.lr_backbone},
                {"params": self.encoder.parameters(),             "lr": self.hparams.lr_backbone},
            ]
        head_params = {"params": self.mdn.parameters(), "lr": self.hparams.lr_head}

        optim = torch.optim.AdamW(bb_params + [head_params],
                                  weight_decay=self.hparams.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optim, T_max=self.trainer.max_epochs)
        return {"optimizer": optim, "lr_scheduler": sched}
