#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# src/scripts/trainer_pointjepa.py
# Train the GraspRegressor; Lightning 1.x-compatible.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import os, re, sys, inspect
import hydra, pytorch_lightning as pl, torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers   import WandbLogger

from scripts.checkpoint_utils     import fetch_checkpoint
from scripts.dlrhand2_datamodule  import DLRHand2DataModule
from scripts.grasp_regressor      import GraspRegressor
from scripts.load_backbone        import load_pretrained_backbone


@hydra.main(config_path="../../configs", config_name="train")
def main(cfg: DictConfig) -> None:

    # ── reproducibility
    pl.seed_everything(cfg.train.seed, workers=True)

    # ── resolve / download checkpoints ------------------------------------------------
    os.makedirs(cfg.ckpt.local_dir, exist_ok=True)
    ckpt_file     = os.path.join(cfg.ckpt.local_dir, cfg.ckpt.filename)
    ckpt_path     = fetch_checkpoint(cfg.ckpt.bucket, cfg.ckpt.filename, ckpt_file)

    resume_file = ckpt_path if ckpt_path and os.path.isfile(ckpt_path) else None
    pre_epochs  = 0
    if resume_file:
        m = re.search(r"epoch=(\d+)-", os.path.basename(resume_file))
        pre_epochs = int(m.group(1)) if m else 0

    max_epochs = pre_epochs + cfg.ckpt.extra_epochs
    print(f"⚙️  Resuming at epoch {pre_epochs}; training until {max_epochs}")

    # ── logger ------------------------------------------------------------------------
    wandb_logger = WandbLogger(
        project = cfg.logger.project,
        entity  = cfg.logger.entity,
        name    = cfg.logger.name,
    )

    # ── data --------------------------------------------------------------------------
    dm = DLRHand2DataModule(
        root_dir   = cfg.data.root_dir,
        ssd_cache_dir = cfg.data.ssd_cache_dir,
        batch_size = cfg.data.batch_size,
        num_workers= cfg.data.num_workers,
        num_points = cfg.data.num_points
    )

    # ── model -------------------------------------------------------------------------
    model = GraspRegressor(
        num_points          = cfg.data.num_points,
        tokenizer_groups    = cfg.model.tokenizer_groups,
        tokenizer_group_size= cfg.model.tokenizer_group_size,
        tokenizer_radius    = cfg.model.tokenizer_radius,
        encoder_dim         = cfg.model.encoder_dim,
        encoder_depth       = cfg.model.encoder_depth,
        encoder_heads       = cfg.model.encoder_heads,
        encoder_dropout     = cfg.model.encoder_dropout,
        encoder_attn_dropout= cfg.model.encoder_attn_dropout,
        encoder_drop_path_rate=cfg.model.encoder_drop_path_rate,
        pooling_type        = cfg.model.pooling_type,
        pooling_heads       = cfg.model.pooling_heads,
        head_hidden_dims    = cfg.model.head_hidden_dims,
        head_output_dim     = cfg.model.head_output_dim,
        grasp_dim           = cfg.model.grasp_dim,
        lr_backbone         = cfg.model.lr_backbone,
        lr_head             = cfg.model.lr_head,
    )

    # debug
    print(
        "### DEBUG:", inspect.getfile(model.__class__),
        "has training_step ->", hasattr(model, "training_step"),
        file=sys.stderr,
    )

    # optional backbone fine-tune ------------------------------------------------------
    if cfg.ckpt.get("backbone_filename"):
        bb_local   = os.path.join(cfg.ckpt.local_dir, cfg.ckpt.backbone_filename)
        bb_ckpt    = fetch_checkpoint(cfg.ckpt.bucket, cfg.ckpt.backbone_filename, bb_local)
        load_pretrained_backbone(model, bb_ckpt)

    # ── trainer -----------------------------------------------------------------------
    trainer = pl.Trainer(
        logger              = wandb_logger,
        accelerator         = cfg.trainer.accelerator,
        devices             = cfg.trainer.devices,
        precision           = cfg.trainer.precision,
        max_epochs          = max_epochs,
        default_root_dir    = cfg.trainer.default_root_dir,
        callbacks           = [LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps   = cfg.trainer.log_every_n_steps,
        gradient_clip_val   = cfg.trainer.get("gradient_clip_val", 0.0),
    )

    # ── run ----------------------------------------------------------------------------
    trainer.fit(model, datamodule=dm, ckpt_path=resume_file)


if __name__ == "__main__":
    main()
