#!/usr/bin/env python3
"""
JointRegressor – predicts the 12-D finger configuration given
  • an object point cloud
  • a 7-D hand pose (translation + quaternion)

The class now supports Point-JEPA fine-tuning:
  • backbone starts frozen and is unfrozen at `encoder_unfreeze_epoch`
  • separate LR groups for backbone vs. head
  • Linear-warm-up + cosine LR schedule (pl-bolts)
  • learnable scalar on positional encodings
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math


from modules.tokenizer   import PointcloudTokenizer
from modules.transformer import TransformerEncoder
from scripts.pooling     import get_pooling



class JointRegressor(pl.LightningModule):
    # ----------------------------------------------------- init
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
        pooling_dropout: float,
        head_hidden_dims: list[int],
        pose_dim: int = 7,
        lr_backbone: float = 1e-4,
        lr_head: float = 3e-3,
        weight_decay: float = 1e-2,
        encoder_unfreeze_epoch: int = 0,
        num_hyp: int = 8,                            # ★ NEW  (K)
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["head_hidden_dims"])

        self.num_hyp = num_hyp                      # ★ NEW
        self.lr_backbone = lr_backbone
        self.lr_head     = lr_head
        self.weight_decay = weight_decay
        self.encoder_unfreeze_epoch = encoder_unfreeze_epoch

        # ---------------- tokenizer --------------------------
        self.tokenizer = PointcloudTokenizer(
            num_groups   = tokenizer_groups,
            group_size   = tokenizer_group_size,
            group_radius = tokenizer_radius,
            token_dim    = encoder_dim,
        )

        # -------------- positional encoding ------------------
        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, encoder_dim),
        )
        self.pe_scale = nn.Parameter(torch.tensor(0.3))  # learnable

        # ---------------- transformer backbone ---------------
        dpr = torch.linspace(0, encoder_drop_path_rate, encoder_depth).tolist()
        self.encoder = TransformerEncoder(
            embed_dim       = encoder_dim,
            depth           = encoder_depth,
            num_heads       = encoder_heads,
            mlp_ratio       = encoder_mlp_ratio,
            qkv_bias        = True,
            drop_rate       = encoder_dropout,
            attn_drop_rate  = encoder_attn_dropout,
            drop_path_rate  = dpr,
            add_pos_at_every_layer = True,
        )

        # ------------------- pooling -------------------------
        self.pool = get_pooling(
            pooling_type, dim=encoder_dim,
            num_heads=pooling_heads,
            dropout=pooling_dropout,
        )

        # ------------------- head --------------------------
        in_dim = encoder_dim + pose_dim
        dims   = [in_dim] + head_hidden_dims + [12 * num_hyp]   # ★ CHANGED
        mlp: list[nn.Module] = []
        for i in range(len(dims) - 2):
            mlp.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        mlp.append(nn.Linear(dims[-2], dims[-1]))
        self.head = nn.Sequential(*mlp)

    # ------------------------------------------------ forward
    @torch.cuda.amp.autocast(enabled=True)
    def forward(self, points: torch.Tensor, pose_vec: torch.Tensor) -> torch.Tensor:
        tokens, centers = self.tokenizer(points)
        pos   = self.positional_encoding(centers) * self.pe_scale
        feats = self.encoder(tokens, pos).last_hidden_state
        obj_emb = self.pool(feats)
        fused   = torch.cat([obj_emb, pose_vec], dim=-1)
        pred    = self.head(fused)                       # (B, 12*K)
        return pred.view(-1, self.num_hyp, 12)           # ★ CHANGED -> (B,K,12)

    # ------------------------------------------- train / val
    def _step(self, batch, stage: str):
        pred  = self(batch["points"], batch["pose"])     # (B,K,12)
        gt    = batch["joints"].unsqueeze(1)             # (B,1,12)

        per_h = (pred - gt).pow(2).mean(-1)              # (B,K)
        loss  = per_h.min(-1).values.mean()              # ★ NEW Min-of-K

        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=gt.size(0))
        return loss

    def training_step(self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): self._step(batch, "val")

    # ----------------------------------------------- optimiser
    def configure_optimizers(self):
        backbone_params = (
            list(self.tokenizer.parameters()) +
            list(self.positional_encoding.parameters()) +
            list(self.encoder.parameters())
        )
        head_params = list(self.pool.parameters()) + list(self.head.parameters())

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.lr_backbone},
                {"params": head_params,     "lr": self.hparams.lr_head},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        # simple cosine schedule with warm-up via LambdaLR
        def lr_lambda(step):
            warmup_steps = 10 * self.trainer.num_training_batches
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # after warmup, cosine decay
            progress = (step - warmup_steps) / float(
                max(1, self.trainer.max_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
    
    
    # ---------- DEBUG SECTION ----------
    def on_after_backward(self) -> None:
        """Runs every time Lightning has done loss.backward()."""
        if self.global_step > 0:                      # only once
            return

        bb_nonzero, bb_total = 0, 0
        for p in self.encoder.parameters():
            if p.requires_grad:
                bb_total += 1
                if p.grad is not None and p.grad.abs().sum() > 0:
                    bb_nonzero += 1
        self.print(f"[GRAD-CHK] encoder tensors with grad: {bb_nonzero}/{bb_total}")

    def on_train_batch_end(self, *_):
        """Verify at least one encoder weight changed after opt.step()."""
        if self.global_step == 0:
            # cache a copy of weights after first backward
            self._enc_before = [
                p.detach().clone() for p in self.encoder.parameters()
                if p.requires_grad
            ]
        elif self.global_step == 1 and hasattr(self, "_enc_before"):
            delta = sum(
                (after - before).abs().sum().item()
                for before, after in zip(self._enc_before, self.encoder.parameters())
                if after.requires_grad
            )
            self.print(f"[WEIGHT-CHK] Σ|Δw_encoder| after 1 step = {delta:.3e}")
            del self._enc_before            # keep RAM clean
    # ---------- END DEBUG ----------


    # ------------------------------------------- unfreeze hook
    def on_train_start(self) -> None:
        if self.hparams.encoder_unfreeze_epoch > 0:
            for m in (self.tokenizer, self.positional_encoding, self.encoder):
                m.requires_grad_(False)

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.hparams.encoder_unfreeze_epoch:
            self.print(">> Unfreezing backbone")
            for m in (self.tokenizer, self.positional_encoding, self.encoder):
                m.requires_grad_(True)

