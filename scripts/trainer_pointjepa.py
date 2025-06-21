#!/usr/bin/env python3
import os, re, glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from checkpoint_utils import fetch_checkpoint
from models import PointJepa
from callbacks.log_at_best_val import TrackLinearAccAtMinLossCallback

import hydra
from omegaconf import DictConfig

class NPZDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        files = glob.glob(os.path.join(self.data_dir, "*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")
        self.files = files

    def train_dataloader(self):
        class D(Dataset):
            def __init__(self, files): self.files = files
            def __len__(self): return len(self.files)
            def __getitem__(self, idx):
                arr = np.load(self.files[idx])
                pts = arr[arr.files[0]]  # expect (N,3)
                return torch.from_numpy(pts).float(), 0

        return DataLoader(D(self.files),
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

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

    # ─── WandB logger ───────────────────────────────────────────────────────
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=cfg.logger.name,
    )

    # ─── Instantiate model & datamodule ────────────────────────────────────
    model = PointJepa(**cfg.model)
    dm = NPZDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # ─── Callbacks ─────────────────────────────────────────────────────────
    cbs = [
        LearningRateMonitor(logging_interval="epoch"),
        TrackLinearAccAtMinLossCallback(),
    ]

    # ─── Trainer & fit ─────────────────────────────────────────────────────
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=max_epochs,
        default_root_dir=cfg.trainer.default_root_dir,
        resume_from_checkpoint=ckpt_path,
        callbacks=cbs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
