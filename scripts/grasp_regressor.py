# scripts/grasp_regressor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models import PointJepa
from scripts.pooling import get_pooling

class GraspRegressor(pl.LightningModule):
    """
    Fine-tunes a Point-JEPA encoder to regress grasp quality scores.
    Pools JEPA tokens, concatenates grasp vectors, then applies an MLP head.
    """
    def __init__(
        self,
        encoder_cfg: dict,
        pooling_type: str = 'mean',
        pooling_heads: int = 4,
        head_hidden_dims: list = [128, 64, 32],
        head_output_dim: int = 1,
        grasp_dim: int = 19,
        lr_backbone: float = 1e-5,
        lr_head: float = 1e-4
    ):
        super().__init__()
        # Pretrained JEPA backbone
        self.encoder = PointJepa(**encoder_cfg)
        # Determine feature dimension D from encoder
        D = getattr(self.encoder.student, 'embed_dim', None) or getattr(self.encoder.student, 'hidden_size', None)
        # Pooling layer
        self.pool = get_pooling(pooling_type, dim=D, num_heads=pooling_heads)
        # Build MLP head input dim = D + grasp_dim
        input_dim = D + grasp_dim
        dims = [input_dim] + head_hidden_dims + [head_output_dim]
        layers = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.head = nn.Sequential(*layers)
        # Learning rates
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head

    def forward(self, x: torch.Tensor, grasp_vec: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 3), grasp_vec: (B, grasp_dim)
        # Encode: obtain patch tokens (B, L, D)
        tokens = self.encoder.student(self.encoder.tokenizer(x))
        # Pool to (B, D)
        obj_feat = self.pool(tokens)
        # Concatenate grasp vector
        combined = torch.cat([obj_feat, grasp_vec], dim=-1)
        # Regress scalar score
        score = self.head(combined).squeeze(-1)
        return score

    def training_step(self, batch, batch_idx):
        pc, grasp_vec, score_true = batch
        score_pred = self(pc, grasp_vec)
        loss = F.smooth_l1_loss(score_pred, score_true)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pc, grasp_vec, score_true = batch
        score_pred = self(pc, grasp_vec)
        loss = F.smooth_l1_loss(score_pred, score_true)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW([
            {'params': self.encoder.parameters(), 'lr': self.lr_backbone},
            {'params': self.head.parameters(),     'lr': self.lr_head},
        ])