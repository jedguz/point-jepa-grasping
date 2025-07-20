#!/usr/bin/env python3
"""
JointRegressor – predicts the 12‑D hand‑joint configuration from
 • an object point cloud
 • a 7‑D hand pose (translation + quaternion)

Supports Point‑JEPA fine‑tuning, multi‑hypothesis (k‑best) outputs and an
optional coordinate change that expresses the pose in the object‑centred frame.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from modules.tokenizer   import PointcloudTokenizer
from modules.transformer import TransformerEncoder
from utils               import transforms
from scripts.pooling     import get_pooling


class JointRegressor(pl.LightningModule):
    # --------------------------------------------------------- init
    def __init__(
        self,
        *,
        num_points: int,
        tokenizer_groups: int,
        tokenizer_group_size: int,
        tokenizer_radius: float,
        transformations: list[str],
        coord_change: bool,
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
        num_pred: int = 5,
        loss_type: str = "min_k_logit",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["head_hidden_dims"])

        self.coord_change            = coord_change
        self.num_pred                = num_pred
        self.loss_type               = loss_type
        self.encoder_unfreeze_epoch  = encoder_unfreeze_epoch

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
        self.pe_scale = nn.Parameter(torch.tensor(0.3))

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

        # ------------------- head ----------------------------
        in_dim = encoder_dim + pose_dim

        if   loss_type == "basic":        out_dim = 12
        elif loss_type == "min_k":        out_dim = 12 * num_pred
        elif loss_type == "min_k_logit":  out_dim = (12 + 1) * num_pred   # k·(12+1)
        elif loss_type == "full":         out_dim = (12 + 2) * num_pred   # k·(12+2)
        else:
            raise ValueError(f"Unknown loss type '{loss_type}'.")

        dims = [in_dim] + head_hidden_dims + [out_dim]

                # Print network architecture
        print(f"\n{'='*50}")
        print(f"MLP Head Architecture (loss_type: {self.loss_type})")
        print(f"{'='*50}")
        print(f"Input dimension: {in_dim}")
        print(f"Hidden dimensions: {head_hidden_dims}")
        print(f"Output dimension: {dims[-1]}")
        print(f"\nLayer-by-layer breakdown:")
        print(f"{'-'*50}")

        mlp: list[nn.Module] = []
        for i in range(len(dims) - 2):
            mlp.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU()])
        mlp.append(nn.Linear(dims[-2], dims[-1]))
        self.head = nn.Sequential(*mlp)

        # learnable scale on the logit/KL term
        if loss_type in {"min_k_logit", "full"}:
            self.logit_scale_raw = nn.Parameter(torch.log(torch.tensor(0.1)))

        # -------- data‑space transforms (optional)------------
        if transformations[0] != "none":
            self.train_transformations = transforms.Compose(
                [self.build_transformation(n) for n in transformations]
            )
        else:
            self.train_transformations = transformations

    # convenience property
    @property
    def logit_scale(self) -> torch.Tensor:
        return self.logit_scale_raw.exp().clamp(0.05, 5.0)

    # ------------------------------------------------ forward
    @torch.cuda.amp.autocast(enabled=True)
    def forward(self, points: torch.Tensor, pose_vec: torch.Tensor):
        tokens, centers = self.tokenizer(points)
        pos   = self.positional_encoding(centers) * self.pe_scale
        feats = self.encoder(tokens, pos).last_hidden_state
        obj_emb = self.pool(feats)

        if self.coord_change:
            pose_vec = pose_vec.clone()
            pose_vec[:, :3] -= centers.mean(dim=1)      # shift to object frame

        fused = torch.cat([obj_emb, pose_vec], dim=-1)
        pred  = self.head(fused)                        # (B, out_dim)

        pred_dim = self.num_pred * 12

        if self.loss_type == "basic":
            return pred                                   # (B,12)

        if self.loss_type == "min_k":
            return pred.view(-1, self.num_pred, 12)       # (B,k,12)

        if self.loss_type == "min_k_logit":
            angles = pred[:, :pred_dim].view(-1, self.num_pred, 12)
            logits = pred[:, pred_dim:].view(-1, self.num_pred)
            return angles, logits

        # full
        angles = pred[:, :pred_dim].view(-1, self.num_pred, 12)
        logits = pred[:, pred_dim : pred_dim + self.num_pred]
        scores = pred[:, pred_dim + self.num_pred :].view(-1, self.num_pred)
        return angles, logits, scores

    # -------------------------------------- training / val
    def _step(self, batch, stage: str):
        loss_terms = {}
        B = batch["points"].size(0)

        if self.loss_type == "basic":
            pred = self(batch["points"], batch["pose"])
            loss_terms["angles"] = F.mse_loss(pred, batch["joints"])

        elif self.loss_type == "min_k":
            angles = self(batch["points"], batch["pose"])          # (B,k,12)
            gt = batch["joints"].unsqueeze(1)                      # (B,1,12)
            per_h = (angles - gt).pow(2).mean(-1)                  # (B,k)
            loss_terms["angles"] = per_h.min(-1).values.mean()

        elif self.loss_type == "min_k_logit":
            angles, logits = self(batch["points"], batch["pose"])
            gt = batch["joints"].unsqueeze(1)
            per_h = (angles - gt).pow(2).mean(-1)                  # (B,k)

            soft_tgt  = F.softmax(-per_h, dim=1)                   # (B,k)
            log_probs = F.log_softmax(logits, dim=1)               # (B,k)

            loss_terms["angles"] = per_h.min(-1).values.mean()
            loss_terms["logit"]  = self.logit_scale * F.kl_div(log_probs, soft_tgt,
                                                               reduction="batchmean")

        elif self.loss_type == "full":
            angles, logits, scores = self(batch["points"], batch["pose"])
            gt = batch["joints"].unsqueeze(1)
            per_h = (angles - gt).pow(2).mean(-1)

            min_idx = torch.argmin(per_h, dim=1)
            loss_terms["angles"] = per_h.min(-1).values.mean()
            loss_terms["logit"]  = self.logit_scale * F.cross_entropy(logits, min_idx)
            loss_terms["score"]  = 0.1 * F.mse_loss(
                scores, batch["meta"]["score"].unsqueeze(1).expand(-1, self.num_pred)
            )

        # unified logging
        total = 0.0
        for k, v in loss_terms.items():
            self.log(f"{stage}_loss_{k}", v, prog_bar=True, batch_size=B)
            total += v
        self.log(f"{stage}_loss", total, prog_bar=True, batch_size=B)
        if hasattr(self, "logit_scale_raw"):
            self.log(f"{stage}_logit_scale", self.logit_scale.item(), prog_bar=True)
        return total

    def training_step  (self, batch, _): return self._step(batch, "train")
    def validation_step(self, batch, _): self._step(batch, "val")

    # -------------------------------------- optimiser
    def configure_optimizers(self):
        backbone_params = (
            list(self.tokenizer.parameters()) +
            list(self.positional_encoding.parameters()) +
            list(self.encoder.parameters())
        )
        head_params = list(self.pool.parameters()) + list(self.head.parameters())

        scale_params = [self.pe_scale]
        if hasattr(self, "logit_scale_raw"):
            scale_params.append(self.logit_scale_raw)

        optim = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.lr_backbone},
                {"params": head_params,     "lr": self.hparams.lr_head},
                {"params": scale_params,    "lr": 1e-4},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        # cosine LR with warm‑up
        def lr_lambda(step):
            warmup = 10 * self.trainer.num_training_batches
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, self.trainer.max_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        return {"optimizer": optim,
                "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    # ---------- transformations for point clouds ----------
    @staticmethod
    def build_transformation(name: str) -> transforms.Transform:
        if   name == "subsample":   return transforms.PointcloudSubsampling(1024)
        elif name == "center":      return transforms.PointcloudCentering()
        elif name == "unit_sphere": return transforms.PointcloudUnitSphere()
        elif name == "rotate":      return transforms.PointcloudRotation(dims=[1])
        else: raise RuntimeError(f"No such transformation: {name}")
    # ---------- DEBUG SECTION --------------------------
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
    # ---------- END DEBUG -------------------------------------
    # --------------- freeze / unfreeze hooks --------------
    def on_train_start(self):
        if self.encoder_unfreeze_epoch > 0:
            for m in (self.tokenizer, self.positional_encoding, self.encoder):
                m.requires_grad_(False)

    def on_train_epoch_start(self):
        if self.current_epoch == self.encoder_unfreeze_epoch:
            self.print(">> Unfreezing backbone")
            for m in (self.tokenizer, self.positional_encoding, self.encoder):
                m.requires_grad_(True)
