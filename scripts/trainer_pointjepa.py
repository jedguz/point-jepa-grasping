# scripts/trainer_grasp.py
#!/usr/bin/env python3
import os, re
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from checkpoint_utils import fetch_checkpoint
from scripts.dlrhand2_datamodule import DLRHand2DataModule
from grasp_regressor import GraspRegressor

@hydra.main(config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # ─── Fetch checkpoint ───────────────────────────────────────────────────
    os.makedirs(cfg.ckpt.local_dir, exist_ok=True)
    local_ckpt = os.path.join(cfg.ckpt.local_dir, cfg.ckpt.filename)
    ckpt_path = fetch_checkpoint(cfg.ckpt.bucket, cfg.ckpt.filename, local_ckpt)

    # ─── Compute total epochs ───────────────────────────────────────────────
    m = re.search(r"epoch=(\d+)-", cfg.ckpt.filename)
    pretrained_epoch = int(m.group(1)) if m else 0
    max_epochs = pretrained_epoch + cfg.ckpt.extra_epochs
    print(f"⚙️  Loaded checkpoint at epoch {pretrained_epoch}, training until {max_epochs}")

    # ─── Setup WandB logger ─────────────────────────────────────────────────
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.logger.name,
    )

    # ─── Instantiate DataModule ─────────────────────────────────────────────
    dm = DLRHand2DataModule(
        root_dir=cfg.data.root_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        num_points=cfg.data.num_points,
    )

    # ─── Instantiate Model ──────────────────────────────────────────────────
    model = GraspRegressor(
        encoder_cfg=cfg.model.encoder,
        pooling_type=cfg.model.pooling.type,
        pooling_heads=cfg.model.pooling.num_heads,
        head_hidden_dims=cfg.head.dims,
        head_output_dim=cfg.head.output_dim,
        grasp_dim=cfg.data.grasp_dim,
        lr_backbone=cfg.optim.lr_backbone,
        lr_head=cfg.optim.lr_head,
    )
    # load pretrained weights
    model.encoder.load_state_dict(
        torch.load(ckpt_path)['state_dict'], strict=False
    )

    # ─── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [LearningRateMonitor(logging_interval="epoch")]

    # ─── Trainer & Fit ─────────────────────────────────────────────────────
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=max_epochs,
        default_root_dir=cfg.trainer.default_root_dir,
        resume_from_checkpoint=None,
        callbacks=callbacks,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
