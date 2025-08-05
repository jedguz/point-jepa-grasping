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
from torchmetrics import Accuracy

from pytorch_lightning.loggers import WandbLogger
from typing import List, Optional, Tuple
from modules.tokenizer   import PointcloudTokenizer
from modules.transformer import TransformerEncoder
from utils               import transforms
from typing import Optional 


class Classification(pl.LightningModule):
    # --------------------------------------------------------- init
    def __init__(
        self,
        num_points: int = 1024,
        tokenizer_num_groups: int = 64,
        tokenizer_group_size: int = 32,
        tokenizer_group_radius: float | None = None,
        tokenizer_unfreeze_epoch: int = 0,
        positional_encoding_unfreeze_epoch: int = 0,
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_heads: int = 6,
        encoder_dropout: float = 0,
        encoder_attention_dropout: float = 0,
        encoder_drop_path_rate: float = 0.2,
        encoder_add_pos_at_every_layer: bool = True,
        encoder_qkv_bias: bool = True,
        encoder_freeze_layers: Optional[List[int]] = None,
        encoder_unfreeze_epoch: int = 0,
        encoder_unfreeze_layers: Optional[List[int]] = None,
        encoder_unfreeze_stepwise: bool = False,
        encoder_unfreeze_stepwise_num_layers: int = 2,
        encoder_learning_rate: Optional[float] = None,
        cls_head: str = "mlp",  # mlp, linear
        cls_head_dim: int = 256,
        cls_head_dropout: float = 0.5,
        cls_head_pooling: str = "mean+max",  # mean+max+cls_token, mean+max, mean, max, cls_token
        loss_label_smoothing: float = 0.2,
        learning_rate: float = 0.001,
        optimizer_adamw_weight_decay: float = 0.05,
        lr_scheduler_linear_warmup_epochs: int = 10,
        lr_scheduler_linear_warmup_start_lr: float = 1e-6,
        lr_scheduler_cosine_eta_min: float = 1e-6,
        pretrained_ckpt_path: str | None = None,
        pretrained_ckpt_ignore_encoder_layers: List[int] = [],
        train_transformations: List[str] = [
            "center",
            "unit_sphere",
        ],  # scale, center, unit_sphere, rotate, translate, height_norm
        val_transformations: List[str] = ["center", "unit_sphere"],
        transformation_scale_min: float = 0.8,
        transformation_scale_max: float = 1.2,
        transformation_scale_symmetries: Tuple[int, int, int] = (1, 0, 1),
        transformation_rotate_dims: List[int] = [1],
        transformation_rotate_degs: Optional[int] = None,
        transformation_translate: float = 0.2,
        transformation_height_normalize_dim: int = 1,
        log_tsne: bool = False,
        log_confusion_matrix: bool = False,
        vote: bool = False,
        vote_num: int = 10,
        encoder_unfreeze_step: int=0,
    ) -> None:
        
        super().__init__()
        self.save_hyperparameters()

        self.encoder_unfreeze_step  = encoder_unfreeze_step

        self.train_acc = None
        self.val_acc = None
        self.vote = vote
        self.val_top_3_acc = None

        def build_transformation(name: str) -> transforms.Transform:
            if name == "scale":
                return transforms.PointcloudScaling(
                    min=transformation_scale_min, max=transformation_scale_max
                )
            elif name == "center":
                return transforms.PointcloudCentering()
            elif name == "unit_sphere":
                return transforms.PointcloudUnitSphere()
            elif name == "rotate":
                return transforms.PointcloudRotation(
                    dims=transformation_rotate_dims, deg=transformation_rotate_degs
                )
            elif name == "translate":
                return transforms.PointcloudTranslation(transformation_translate)
            elif name == "height_norm":
                return transforms.PointcloudHeightNormalization(
                    transformation_height_normalize_dim
                )
            else:
                raise RuntimeError(f"No such transformation: {name}")

        self.train_transformations = transforms.Compose(
            [transforms.PointcloudSubsampling(num_points)]
            + [build_transformation(name) for name in train_transformations]
        )
        self.val_transformations = transforms.Compose(
            [transforms.PointcloudSubsampling(num_points)]
            + [build_transformation(name) for name in val_transformations]
        )

        # ---------------- tokenizer --------------------------
        self.tokenizer = PointcloudTokenizer(
            num_groups   = tokenizer_num_groups,
            group_size   = tokenizer_group_size,
            group_radius = tokenizer_group_radius,
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
            mlp_ratio       = 4,
            qkv_bias        = True,
            drop_rate       = encoder_dropout,
            attn_drop_rate  = 0.1,
            drop_path_rate  = dpr,
            add_pos_at_every_layer = True,
        )

        # ------------------- head ----------------------------
        match cls_head_pooling:
            case "cls_token" | "mean+max+cls_token":
                init_std = 0.02
                self.cls_token = nn.Parameter(torch.zeros(encoder_dim))
                nn.init.trunc_normal_(
                    self.cls_token, mean=0, std=init_std, a=-init_std, b=init_std
                )
                self.cls_pos = nn.Parameter(torch.zeros(encoder_dim))
                nn.init.trunc_normal_(
                    self.cls_pos, mean=0, std=init_std, a=-init_std, b=init_std
                )
                self.first_cls_head_dim = (
                    encoder_dim if cls_head_pooling == "cls_token" else 3 * encoder_dim
                )
            case "mean+max":
                self.first_cls_head_dim = 2 * encoder_dim
            case "mean":
                self.first_cls_head_dim = encoder_dim
            case "max":
                self.first_cls_head_dim = encoder_dim
            case _:
                raise ValueError()

        self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.2)


    def setup(self, stage: Optional[str] = None) -> None:
        self.train_acc = Accuracy(task="multiclass", num_classes=41)  # TODO: Need num_classes here
        self.val_acc = Accuracy(task="multiclass", num_classes=41)  # TODO: Need num_classes here
        if self.vote:
            self.val_vote_acc = Accuracy(task="multiclass", num_classes=41)
        self.val_top_3_acc = Accuracy(
            top_k=3, task="multiclass", num_classes=41)
        if self.hparams.cls_head == "mlp":  # type: ignore
            self.cls_head = nn.Sequential(
                nn.Linear(
                    self.first_cls_head_dim, self.hparams.cls_head_dim, bias=False  # type: ignore
                ),  # bias can be False because of batch norm following
                nn.BatchNorm1d(self.hparams.cls_head_dim),  # type: ignore
                nn.ReLU(),
                nn.Dropout(self.hparams.cls_head_dropout),  # type: ignore
                nn.Linear(self.hparams.cls_head_dim, self.hparams.cls_head_dim, bias=False),  # type: ignore
                nn.BatchNorm1d(self.hparams.cls_head_dim),  # type: ignore
                nn.ReLU(),
                nn.Dropout(self.hparams.cls_head_dropout),  # type: ignore
                nn.Linear(self.hparams.cls_head_dim, 41),  # type: ignore
            )
        elif self.hparams.cls_head == "linear":  # type: ignore
            self.cls_head = nn.Linear(self.first_cls_head_dim, 41)  # type: ignore

        self.val_macc = Accuracy(num_classes=41, average="macro", task="multiclass")  # type: ignore

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.watch(self)
                logger.experiment.define_metric("val_acc", summary="last,max")
                logger.experiment.define_metric("val_top_3_acc", summary="last,max")
                logger.experiment.define_metric("val_macc", summary="last,max")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: (B, N, 3)
        tokens: torch.Tensor  # (B, T, C)
        centers: torch.Tensor  # (B, T, 3)
        tokens, centers = self.tokenizer(points)
        points = self.train_transformations(points)
        pos_embeddings = self.positional_encoding(centers)
        if self.hparams.cls_head_pooling in ["cls_token", "mean+max+cls_token"]:  # type: ignore
            B, T, C = tokens.shape
            tokens = torch.cat(
                [self.cls_token.reshape(1, 1, C).expand(B, -1, -1), tokens], dim=1
            )
            pos_embeddings = torch.cat(
                [self.cls_pos.reshape(1, 1, C).expand(B, -1, -1), pos_embeddings], dim=1
            )
        tokens = self.encoder(tokens, pos_embeddings).last_hidden_state
        match self.hparams.cls_head_pooling:  # type: ignore
            case "cls_token":
                embedding = tokens[:, 0]
            case "mean+max":
                max_features = torch.max(tokens, dim=1).values
                mean_features = torch.mean(tokens, dim=1)
                embedding = torch.cat([max_features, mean_features], dim=-1)
            case "mean+max+cls_token":
                cls_token = tokens[:, 0]
                max_features = torch.max(tokens[:, 1:], dim=1).values
                mean_features = torch.mean(tokens[:, 1:], dim=1)
                embedding = torch.cat([cls_token, max_features, mean_features], dim=-1)
            case "mean":
                embedding = torch.mean(tokens, dim=1)
            case "max":
                embedding = torch.max(tokens, dim=1).values
            case _:
                raise ValueError(f"Unknown cls_head_pooling: {self.hparams.cls_head_pooling}")  # type: ignore
        logits = (
            self.cls_head(embedding)  # type: ignore
            if isinstance(tokens, torch.Tensor)
            else self.cls_head(embedding.F)  # type: ignore
        )
        return logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        # points: (B, N, 3)
        # label: (B,)
        points = batch["points"]
        label = batch["labels"]
        logits = self.forward(points)

        loss = self.loss_func(logits, label)
        self.log("train_loss", loss, on_epoch=True)

        pred = torch.max(logits, dim=-1).indices
        self.train_acc(pred, label)
        self.log("train_acc", self.train_acc, on_epoch=True)

        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        # points: (B, N, 3)
        # label: (B,)
        points = batch["points"]
        label = batch["labels"]
        points = self.val_transformations(points)
        logits = self.forward(points)

        loss = self.loss_func(logits, label)
        self.log("val_loss", loss)

        pred = torch.max(logits, dim=-1).indices
        self.val_acc(pred, label)
        self.log("val_acc", self.val_acc)
        self.val_top_3_acc(logits, label)
        self.log("val_top_3_acc", self.val_top_3_acc)
        self.val_macc(pred, label)
        self.log("val_macc", self.val_macc)

        if self.hparams.vote:  # type: ignore
            logits_list = []
            for _ in range(self.hparams.vote_num):  # type: ignore
                points = self.val_transformations(points)
                logits = self.forward(points)
                logits_list.append(F.softmax(logits, dim=-1))
            mean_logits = torch.mean(torch.stack(logits_list), dim=0)
            vote_pred = torch.max(mean_logits, dim=-1).indices

            self.val_vote_acc(vote_pred, label)
            self.log("val_vote_acc", self.val_vote_acc)

    # -------------------------------------- optimiser
    def configure_optimizers(self):
        assert self.trainer is not None

        encoder_params = []
        other_params = []
        for name, param in self.named_parameters():
            if name.startswith("encoder."):
                encoder_params.append(param)
            else:
                other_params.append(param)

        enc_lr: Optional[float] = self.hparams.encoder_learning_rate  # type: ignore
        opt = torch.optim.AdamW(
            params=[
                {"params": encoder_params, "lr": enc_lr if enc_lr is not None else self.hparams.learning_rate},  # type: ignore
                {"params": other_params},
            ],
            lr=self.hparams.learning_rate,  # type: ignore
            weight_decay=self.hparams.optimizer_adamw_weight_decay,  # type: ignore
        )

        def lr_lambda(step):
            warmup = 10 * self.trainer.num_training_batches
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, self.trainer.max_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        return [opt], [sched]
    
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
        if self.encoder_unfreeze_step > 0:
            for m in (self.tokenizer, self.positional_encoding, self.encoder):
                m.requires_grad_(False)

    """ EPOCH -> NOW STEP
    def on_train_epoch_start(self):
        if self.current_epoch == self.encoder_unfreeze_epoch:
            self.print(f">> Unfreezing encoder at epoch {self.current_epoch}")
            for m in (self.tokenizer, self.positional_encoding, self.encoder):
                m.requires_grad_(True)
    """

    def on_train_batch_start(self, batch, batch_idx):
        if self.global_step == self.encoder_unfreeze_step:
            self.print(f">> Unfreezing encoder at step {self.global_step}")
            for m in (self.tokenizer, self.positional_encoding, self.encoder):
                m.requires_grad_(True)
