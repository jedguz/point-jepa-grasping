#!/usr/bin/env python3
# scripts/train_pointjepa.py
"""
Training entry-point for the GraspRegressor.

Notable fixes
-------------
* Resume only when a checkpoint is actually present.
* Correct max_epochs calculation for the fresh-run case.
* `pl.seed_everything` for reproducibility.
* Dropped the manual `dm.setup()` (Lightning calls it).
"""
from __future__ import annotations

import os
import re

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from checkpoint_utils import fetch_checkpoint
from dlrhand2_datamodule import DLRHand2DataModule
from grasp_regressor import GraspRegressor
from load_backbone   import load_pretrained_backbone


@hydra.main(config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:  # pragma: no cover
    # ── seeding
    pl.seed_everything(cfg.train.seed, workers=True)

    # ── checkpoint handling
    os.makedirs(cfg.ckpt.local_dir, exist_ok=True)
    local_ckpt = os.path.join(cfg.ckpt.local_dir, cfg.ckpt.filename)
    ckpt_path: str | None = fetch_checkpoint(
        cfg.ckpt.bucket, cfg.ckpt.filename, local_ckpt
    )

    pre_epochs = 0
    if ckpt_path:
        m = re.search(r"epoch=(\d+)-", os.path.basename(ckpt_path))
        pre_epochs = int(m.group(1)) if m else 0

    max_epochs = pre_epochs + cfg.ckpt.extra_epochs
    print(f"⚙️  Resuming at epoch {pre_epochs}; training until {max_epochs}")

    # ── logger
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.logger.name,
    )

    # ── data
    dm = DLRHand2DataModule(
        root_dir=cfg.data.root_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        num_points=cfg.data.num_points,
    )

    # ── model
    model = GraspRegressor(
        num_points=cfg.data.num_points,
        tokenizer_groups=cfg.model.tokenizer_groups,
        tokenizer_group_size=cfg.model.tokenizer_group_size,
        tokenizer_radius=cfg.model.tokenizer_radius,
        encoder_dim=cfg.model.encoder_dim,
        encoder_depth=cfg.model.encoder_depth,
        encoder_heads=cfg.model.encoder_heads,
        encoder_dropout=cfg.model.encoder_dropout,
        encoder_attn_dropout=cfg.model.encoder_attn_dropout,
        encoder_drop_path_rate=cfg.model.encoder_drop_path_rate,
        pooling_type=cfg.model.pooling_type,
        pooling_heads=cfg.model.pooling_heads,
        head_hidden_dims=cfg.model.head_hidden_dims,
        head_output_dim=cfg.model.head_output_dim,
        grasp_dim=cfg.model.grasp_dim,
        lr_backbone=cfg.model.lr_backbone,
        lr_head=cfg.model.lr_head,
    )

    # ── Load ShapeNet-pre-trained backbone ───────────────────────────────
    if cfg.ckpt.get("backbone_filename"):
        local_backbone = os.path.join(
            cfg.ckpt.local_dir, cfg.ckpt.backbone_filename
        )
        backbone_ckpt = fetch_checkpoint(
            cfg.ckpt.bucket, cfg.ckpt.backbone_filename, local_backbone
        )
        load_pretrained_backbone(model, backbone_ckpt)

    # ── callbacks
    callbacks = [LearningRateMonitor(logging_interval="epoch")]

    # ── trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=max_epochs,
        default_root_dir=cfg.trainer.default_root_dir,
        resume_from_checkpoint=ckpt_path if ckpt_path else None,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", 0.0),
    )

    # ── run
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
