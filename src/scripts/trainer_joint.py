#!/usr/bin/env python3
from __future__ import annotations
import os, sys
import hydra, pytorch_lightning as pl, torch
torch.set_float32_matmul_precision('medium')

from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar


from scripts.dlrhand2_joint_datamodule import DLRHand2JointDataModule
from scripts.joint_regressor import JointRegressor
from scripts.checkpoint_utils import fetch_checkpoint, load_full_checkpoint
from callbacks.backbone_embedding_inspector import BackboneEmbeddingInspector
from callbacks.additional_metrics_callbacks import AdditionalMetrics
from callbacks.save_metrics_csv import SaveMetricsCSV 

@hydra.main(version_base="1.1", config_path="../../configs", config_name="train_joint")
def main(cfg: DictConfig) -> None:
    print("\n ---START TRAINING--: \n")
    pl.seed_everything(cfg.train.seed, workers=True)

    # ────────────────────────────────────────────────────────────────────────
    # Full-run checkpoint resume (optional)
    # ────────────────────────────────────────────────────────────────────────
    os.makedirs(cfg.ckpt.local_dir, exist_ok=True)
    resume = None
    if cfg.ckpt.filename:
        ckpt_file = os.path.join(cfg.ckpt.local_dir, cfg.ckpt.filename)
        ckpt_path = fetch_checkpoint(cfg.ckpt.bucket, cfg.ckpt.filename, ckpt_file)
        if ckpt_path and os.path.isfile(ckpt_path):
            resume = ckpt_path
            print(f">> Resuming training from checkpoint: {cfg.ckpt.filename}")
        else:
            print(f">> Checkpoint {cfg.ckpt.filename} not found, starting fresh run.")

    # ────────────────────────────────────────────────────────────────────────
    # Data Module
    # ────────────────────────────────────────────────────────────────────────
    print("\n ---INITIATE DATA MODULE---: \n")
    dm = DLRHand2JointDataModule(
        root_dir      = cfg.data.root_dir,
        ssd_cache_dir = cfg.data.ssd_cache_dir,
        batch_size    = cfg.data.batch_size,
        num_workers   = cfg.data.num_workers,
        num_points    = cfg.data.num_points,
        score_temp    = cfg.data.score_temp,
        split_file    = cfg.data.split_file,
        preload_all   = cfg.data.preload_all,
    )

    # ────────────────────────────────────────────────────────────────────────
    # Model Initialization
    # ────────────────────────────────────────────────────────────────────────
    print("\n ---INITIATE THE REGRESSION TRAINING PIPELINE---: \n")
    model = JointRegressor(
        num_points             = cfg.data.num_points,
        tokenizer_groups       = cfg.model.tokenizer_groups,
        tokenizer_group_size   = cfg.model.tokenizer_group_size,
        tokenizer_radius       = cfg.model.tokenizer_radius,
        transformations        = cfg.model.transformations,
        coord_change           = cfg.model.coord_change,
        encoder_dim            = cfg.model.encoder_dim,
        encoder_depth          = cfg.model.encoder_depth,
        encoder_heads          = cfg.model.encoder_heads,
        encoder_dropout        = cfg.model.encoder_dropout,
        encoder_attn_dropout   = cfg.model.encoder_attn_dropout,
        encoder_drop_path_rate = cfg.model.encoder_drop_path_rate,
        encoder_mlp_ratio      = cfg.model.encoder_mlp_ratio,
        pooling_type           = cfg.model.pooling_type,
        pooling_heads          = cfg.model.pooling_heads,
        pooling_dropout        = 0.1,
        head_hidden_dims       = cfg.model.head_hidden_dims,
        head_dropout           = cfg.model.head_dropout,  # ← NEW
        pose_dim               = 7,
        lr_backbone            = cfg.model.lr_backbone,
        lr_head                = cfg.model.lr_head,
        weight_decay           = cfg.model.weight_decay,
        encoder_unfreeze_epoch = cfg.model.encoder_unfreeze_epoch,
        encoder_unfreeze_step  = cfg.model.encoder_unfreeze_step,
        num_pred               = cfg.model.num_pred,
        loss_type              = cfg.model.loss_type,
        logit_scale_init       = cfg.model.logit_scale_init,
        logit_scale_min        = cfg.model.logit_scale_min,
        logit_scale_max        = cfg.model.logit_scale_max,
    )

    # ─────────────────────────────────────────────────────────────
    # Backbone checkpoint loading (strongly verified)
    # ─────────────────────────────────────────────────────────────
    import hashlib   # <-- top of file if not already imported

    print("\n ---BACKBONE CHECKPOINT HANDLING (HARDENED)---: \n")
    load_bb = bool(cfg.ckpt.get("load_backbone", False))

    # 1. Mutual-exclusion guard
    bb_filename = cfg.ckpt.get("backbone_filename", "").strip()
    if load_bb is False and bb_filename:
        raise ValueError(
            "cfg.ckpt.load_backbone is False **but** backbone_filename is set "
            f"({bb_filename}). Either set load_backbone=True or clear the filename."
        )
    if load_bb is True and not bb_filename:
        raise ValueError(
            "cfg.ckpt.load_backbone is True but backbone_filename is empty."
        )

    if load_bb:
        bb_local = os.path.join(cfg.ckpt.local_dir, bb_filename)
        bb_ckpt = fetch_checkpoint(cfg.ckpt.bucket, bb_filename, bb_local)
        if not bb_ckpt or not os.path.isfile(bb_ckpt):
            raise FileNotFoundError(
                f"Backbone checkpoint '{bb_filename}' not found in {cfg.ckpt.local_dir}"
            )

        # 2. Optional SHA-256 verification (add your own hash in the config)
        expected_sha256 = cfg.ckpt.get("backbone_sha256", "").strip()
        if expected_sha256:
            with open(bb_ckpt, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash != expected_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {bb_filename}:\n"
                    f"  expected {expected_sha256}\n"
                    f"  actual   {file_hash}"
                )
            print(f">> Checksum OK: {file_hash}")

        # 3. Load the weights
        n_loaded = load_full_checkpoint(model, bb_ckpt)  # <-- change your util to return count
        if n_loaded == 0:
            raise RuntimeError(
                "load_full_checkpoint reported 0 parameters loaded – aborting."
            )
        print(f">> Loaded {n_loaded:,} backbone parameters from {bb_filename}")

        # 4. Log hash & count to WandB for audit
        wandb_logger = None  # you haven't built it yet; stash in cfg and add later
        cfg._loaded_backbone = {
            "filename": bb_filename,
            "sha256": expected_sha256 or file_hash,
            "params_loaded": n_loaded,
        }

    else:
        print(">> Skipping backbone checkpoint (load_backbone=False)")
        cfg._loaded_backbone = {"filename": None}

    # ────────────────────────────────────────────────────────────────────────
    # Trainer setup
    # ────────────────────────────────────────────────────────────────────────
    
    print("\n ---SET UP THE TRAINER---: \n")
    wandb_logger = WandbLogger(
        project = cfg.logger.project,
        entity  = cfg.logger.entity,
        name    = cfg.logger.name,
    )

    wandb_logger.experiment.config.update({"backbone_info": cfg._loaded_backbone})

    # SWITCHED EPOCHS TO STEPS
    #max_epochs = cfg.trainer.max_epochs
    #print(f"⚙️ Training for {max_epochs} epoch(s)")

    # ✅ new
    max_steps = cfg.trainer.max_steps
    print(f"⚙️ Training for {max_steps:,} optimisation steps")

    trainer = pl.Trainer(
        logger             = wandb_logger,
        accelerator        = cfg.trainer.accelerator,
        devices            = cfg.trainer.devices,
        precision          = cfg.trainer.precision,
 #       max_epochs         = max_epochs,
        max_steps          = max_steps,
        default_root_dir   = cfg.trainer.default_root_dir,
        callbacks          = [
            LearningRateMonitor(logging_interval="epoch"),
            # BackboneEmbeddingInspector(num_batches=8),
            AdditionalMetrics(tau_degrees=(10,15)),
            SaveMetricsCSV(),
        ],
        log_every_n_steps  = cfg.trainer.log_every_n_steps,
        gradient_clip_val  = cfg.trainer.get("gradient_clip_val", 1.0),
        overfit_batches    = cfg.trainer.get("overfit_batches", 0),
        val_check_interval      = cfg.trainer.val_check_interval,
        check_val_every_n_epoch = cfg.trainer.check_val_every_n_epoch,
        num_sanity_val_steps    = cfg.trainer.num_sanity_val_steps,  # ← NEW
    )

    # ────────────────────────────────────────────────────────────────────────
    # Start training
    # ────────────────────────────────────────────────────────────────────────
    print("\n ---START FITTING---: \n")
    trainer.fit(model, datamodule=dm, ckpt_path=resume)


if __name__ == "__main__":
    main()
