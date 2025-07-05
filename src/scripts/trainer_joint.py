# src/scripts/trainer_joint.py
#!/usr/bin/env python3
from __future__ import annotations
import os, re, inspect, sys
import copy, math
import hydra, pytorch_lightning as pl, torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers   import WandbLogger

from scripts.dlrhand2_joint_datamodule import DLRHand2JointDataModule
from scripts.joint_regressor           import JointRegressor
from scripts.checkpoint_utils          import fetch_checkpoint
from scripts.checkpoint_utils          import load_full_checkpoint
from callbacks.backbone_embedding_inspector import BackboneEmbeddingInspector


@hydra.main(config_path="../../configs", config_name="train_joint")
def main(cfg: DictConfig) -> None:                                                     
    pl.seed_everything(cfg.train.seed, workers=True)

    # ─ checkpoint handling ───────────────────────────────────────────────────
    os.makedirs(cfg.ckpt.local_dir, exist_ok=True)
    ckpt_file = os.path.join(cfg.ckpt.local_dir, cfg.ckpt.filename)
    ckpt_path = fetch_checkpoint(cfg.ckpt.bucket, cfg.ckpt.filename, ckpt_file)
    resume = ckpt_path if ckpt_path and os.path.isfile(ckpt_path) else None
    pre_ep = int(re.search(r"epoch=(\d+)-", ckpt_path or "") .group(1)) if resume else 0
    max_epochs = pre_ep + cfg.ckpt.extra_epochs
    print(f"⚙️ Resuming at epoch {pre_ep}; training until {max_epochs}")

    # ─ logger ────────────────────────────────────────────────────────────────
    wandb_logger = WandbLogger(
        project = cfg.logger.project,
        entity  = cfg.logger.entity,
        name    = cfg.logger.name,
    )

    # ─ data ──────────────────────────────────────────────────────────────────
    dm = DLRHand2JointDataModule(
        root_dir      = cfg.data.root_dir,
        ssd_cache_dir = cfg.data.ssd_cache_dir,
        batch_size    = cfg.data.batch_size,
        num_workers   = cfg.data.num_workers,
        num_points    = cfg.data.num_points,
        top_percentile = cfg.data.top_percentile,
        sort_by_score = cfg.data.sort_by_score,
    )

    # ─ model ─────────────────────────────────────────────────────────────────
    model = JointRegressor(
        num_points           = cfg.data.num_points,
        tokenizer_groups     = cfg.model.tokenizer_groups,
        tokenizer_group_size = cfg.model.tokenizer_group_size,
        tokenizer_radius     = cfg.model.tokenizer_radius,
        encoder_dim          = cfg.model.encoder_dim,
        encoder_depth        = cfg.model.encoder_depth,
        encoder_heads        = cfg.model.encoder_heads,
        encoder_dropout      = cfg.model.encoder_dropout,
        encoder_attn_dropout = cfg.model.encoder_attn_dropout,
        encoder_drop_path_rate=cfg.model.encoder_drop_path_rate,
        encoder_mlp_ratio    = cfg.model.encoder_mlp_ratio,
        pooling_type         = cfg.model.pooling_type,
        pooling_heads        = cfg.model.pooling_heads,
        pooling_dropout      = 0.1,
        head_hidden_dims     = cfg.model.head_hidden_dims,
        pose_dim             = 7,
        lr_backbone          = cfg.model.lr_backbone,
        lr_head              = cfg.model.lr_head,
    )

    if cfg.ckpt.get("backbone_filename"):
        bb_local = os.path.join(cfg.ckpt.local_dir, cfg.ckpt.backbone_filename)
        bb_ckpt  = fetch_checkpoint(cfg.ckpt.bucket, cfg.ckpt.backbone_filename, bb_local)
        load_full_checkpoint(model, bb_ckpt)

    # ─ trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        logger             = wandb_logger,
        accelerator        = cfg.trainer.accelerator,
        devices            = cfg.trainer.devices,
        precision          = cfg.trainer.precision,
        max_epochs         = max_epochs,
        default_root_dir   = cfg.trainer.default_root_dir,
        callbacks          = [
            LearningRateMonitor(logging_interval="epoch"),
            BackboneEmbeddingInspector(num_batches=8),
        ],
        log_every_n_steps  = cfg.trainer.log_every_n_steps,
        gradient_clip_val  = cfg.trainer.get("gradient_clip_val", 0.0),
    )

    dm.setup()
    trainer.fit(model, datamodule=dm, ckpt_path=resume)


if __name__ == "__main__":
    main()
