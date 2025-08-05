#!/usr/bin/env python3
from __future__ import annotations
import os, sys
import hydra, pytorch_lightning as pl, torch
torch.set_float32_matmul_precision('medium')

from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar


from scripts.dlrhand2_classification_datamodule import DLRHand2JointDataModule
from scripts.classification import Classification
from scripts.checkpoint_utils import fetch_checkpoint, load_full_checkpoint
from callbacks.backbone_embedding_inspector import BackboneEmbeddingInspector
from callbacks.additional_metrics_callbacks import AdditionalMetrics
from callbacks.save_metrics_csv import SaveMetricsCSV 

@hydra.main(version_base="1.1", config_path="../../configs", config_name="train_classification")
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
    model = Classification(
        num_points = cfg.data.num_points,
        tokenizer_num_groups = cfg.model.tokenizer_num_groups,
        tokenizer_group_size = cfg.model.tokenizer_group_size,
        tokenizer_group_radius = cfg.model.tokenizer_group_radius,
        tokenizer_unfreeze_epoch = cfg.model.tokenizer_unfreeze_epoch,
        positional_encoding_unfreeze_epoch = cfg.model.positional_encoding_unfreeze_epoch,
        encoder_dim = cfg.model.encoder_dim,
        encoder_depth = cfg.model.encoder_depth,
        encoder_heads = cfg.model.encoder_heads,
        encoder_dropout = cfg.model.encoder_dropout,
        encoder_attention_dropout = cfg.model.encoder_attention_dropout,
        encoder_drop_path_rate = cfg.model.encoder_drop_path_rate,
        encoder_add_pos_at_every_layer = cfg.model.encoder_add_pos_at_every_layer,
        encoder_qkv_bias = cfg.model.encoder_qkv_bias,
        encoder_freeze_layers = cfg.model.encoder_freeze_layers,
        encoder_unfreeze_epoch = cfg.model.encoder_unfreeze_epoch,
        encoder_unfreeze_layers = cfg.model.encoder_unfreeze_layers,
        encoder_unfreeze_stepwise = cfg.model.encoder_unfreeze_stepwise,
        encoder_unfreeze_stepwise_num_layers = cfg.model.encoder_unfreeze_stepwise_num_layers,
        encoder_learning_rate = cfg.model.encoder_learning_rate,
        cls_head = cfg.model.cls_head,
        cls_head_dim = cfg.model.cls_head_dim,
        cls_head_dropout = cfg.model.cls_head_dropout,
        cls_head_pooling = cfg.model.cls_head_pooling,
        loss_label_smoothing = cfg.model.loss_label_smoothing,
        learning_rate = cfg.model.learning_rate,
        optimizer_adamw_weight_decay = cfg.model.optimizer_adamw_weight_decay,
        lr_scheduler_linear_warmup_epochs = cfg.model.lr_scheduler_linear_warmup_epochs,
        lr_scheduler_linear_warmup_start_lr = cfg.model.lr_scheduler_linear_warmup_start_lr,
        lr_scheduler_cosine_eta_min = cfg.model.lr_scheduler_cosine_eta_min,
        pretrained_ckpt_path = cfg.model.pretrained_ckpt_path,
        pretrained_ckpt_ignore_encoder_layers = cfg.model.pretrained_ckpt_ignore_encoder_layers,
        train_transformations = cfg.model.train_transformations,
        val_transformations = cfg.model.val_transformations,
        transformation_scale_min = cfg.model.transformation_scale_min,
        transformation_scale_max = cfg.model.transformation_scale_max,
        transformation_scale_symmetries = cfg.model.transformation_scale_symmetries,
        transformation_rotate_dims = cfg.model.transformation_rotate_dims,
        transformation_rotate_degs = cfg.model.transformation_rotate_degs,
        transformation_translate = cfg.model.transformation_translate,
        transformation_height_normalize_dim = cfg.model.transformation_height_normalize_dim,
        log_tsne = cfg.model.log_tsne,
        log_confusion_matrix = cfg.model.log_confusion_matrix,
        vote = cfg.model.vote,
        vote_num = cfg.model.vote_num,
        encoder_unfreeze_step = cfg.model.encoder_unfreeze_step,
    )

    # ────────────────────────────────────────────────────────────────────────
    # Backbone checkpoint loading (controlled by boolean flag)
    # ────────────────────────────────────────────────────────────────────────
    print("\n ---BACKBONE CHECKPOINT HANDLING---: \n")
    load_bb = cfg.ckpt.get("load_backbone", False)
    if load_bb:
        # require a valid filename
        bb_filename = cfg.ckpt.get("backbone_filename", "").strip()
        if not bb_filename:
            raise ValueError("load_backbone is True but no backbone_filename provided in config.ckpt.backbone_filename")
        bb_local = os.path.join(cfg.ckpt.local_dir, bb_filename)
        bb_ckpt = fetch_checkpoint(cfg.ckpt.bucket, bb_filename, bb_local)
        # error if missing
        if not bb_ckpt or not os.path.isfile(bb_ckpt):
            raise FileNotFoundError(f"Backbone checkpoint '{bb_filename}' not found in {cfg.ckpt.local_dir}")
        load_full_checkpoint(model, bb_ckpt)
        print(f">> Loaded backbone weights from {bb_filename}")
    else:
        print(">> Skipping backbone checkpoint (load_backbone is False)")

    # PRINT WEIGHT STATISTICS
    for name, param in model.named_parameters():
        print(f"{name}: mean={param.data.mean().item():.5f}, std={param.data.std().item():.5f}")


    # ────────────────────────────────────────────────────────────────────────
    # Trainer setup
    # ────────────────────────────────────────────────────────────────────────
    
    print("\n ---SET UP THE TRAINER---: \n")
    wandb_logger = WandbLogger(
        project = cfg.logger.project,
        entity  = cfg.logger.entity,
        name    = cfg.logger.name,
    )

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
            # AdditionalMetrics(tau_degrees=(10,15)),
            # SaveMetricsCSV(),
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
