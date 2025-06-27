# Call using python training_final.py fit   --data.data_dir=03001627   --data.batch_size=10   --model.predictor_add_target_pos=true   --model.num_targets_per_sample=15

import glob, os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.cli import LightningCLI

from pointjepa.models import PointJepa
from pointjepa.callbacks.log_at_best_val import TrackLinearAccAtMinLossCallback


class LossHistoryCallback(pl.Callback):
    def __init__(self):
        self.losses = []

    def on_train_epoch_end(self, trainer, pl_module, *args):
        # Lightning logs the avg train loss as "train_loss" by default
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.losses.append(loss.cpu().item())

    def on_fit_end(self, trainer, pl_module, *args):
        import matplotlib.pyplot as plt
        out_dir = trainer.default_root_dir
        os.makedirs(out_dir, exist_ok=True)
        plt.figure()
        plt.plot(self.losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Train SSL Loss")
        plt.title("JEPA Train Loss Curve")
        plt.savefig(os.path.join(out_dir, "loss_curve.png"))
        plt.close()


class NPZDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = "03001627",
                 batch_size: int = 10,
                 num_workers: int = 0):
        super().__init__()
        self.data_dir    = data_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        files = glob.glob(os.path.join(self.data_dir, "*.npz"))
        if not files:
            raise FileNotFoundError(f"No .npz files found in {self.data_dir}")
        self.files = files

    def train_dataloader(self):
        class D(Dataset):
            def __init__(self, files):
                self.files = files
            def __len__(self):
                return len(self.files)
            def __getitem__(self, idx):
                arr = np.load(self.files[idx])
                pts = arr[arr.files[0]]  # should be (2048,3)
                return torch.from_numpy(pts).float(), 0

        ds = D(self.files)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    LightningCLI(
        model_class=PointJepa,
        datamodule_class=NPZDataModule,
        trainer_defaults={
            "default_root_dir": "artifacts_npz_simple2",
            "accelerator": "gpu",
            "devices": 1,
            "precision": 16,
            "max_epochs": 30,
            "log_every_n_steps": 5,
            "callbacks": [
                LearningRateMonitor(logging_interval="epoch"),
                TrackLinearAccAtMinLossCallback(),
                LossHistoryCallback(),         # record & save loss curve
            ],
        },
        seed_everything_default=42,
        save_config_callback=None,
    )
