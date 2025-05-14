# src/train/pretrain_module.py  (pseudo‑code)

from torch import nn
import torch.nn.functional as F
from models.voxel_encoder import VoxelEncoder
from common.ema import ModelEmaV2

class PretrainJEPA(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        # student
        self.encoder = VoxelEncoder()
        self.predictor = nn.Sequential(
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim),
            nn.GELU(),
            nn.Linear(self.encoder.out_dim, self.encoder.out_dim)
        )

        # teacher = EMA of student (no grads)
        self.teacher = ModelEmaV2(self.encoder,
                                  decay=self.hparams.get("ema_decay", 0.996))
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    def training_step(self, batch, _):
        grid = batch["voxel"]            # (B,1,48,48,48)
        ctx, tgt = mask_tokens(grid)     # implement your masking
        # encode
        student_lat = self.encoder(ctx)          # (B,512)
        teacher_lat = self.teacher.module(tgt)   # (B,512)
        # predictor head
        pred = self.predictor(student_lat)
        loss = 1 - F.cosine_similarity(pred, teacher_lat.detach()).mean()
        self.log("train/loss", loss)
        return loss
