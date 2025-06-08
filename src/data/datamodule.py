# src/data/datamodule.py
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from .pointcloud_dataset import PointCloudDataset

class PointCloudDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, use_embeddings=False, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_embeddings = use_embeddings
        self.num_workers = num_workers

    def setup(self, stage=None):
        # you could split train/val by folders or give them separate roots
        self.train_ds = PointCloudDataset(self.data_dir, use_embeddings=self.use_embeddings)
        self.val_ds   = PointCloudDataset(self.data_dir, use_embeddings=self.use_embeddings)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
