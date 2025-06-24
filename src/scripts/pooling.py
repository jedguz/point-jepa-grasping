# scripts/pooling.py
import torch
import torch.nn as nn
from torch.nn import functional as F

class MeanPool(nn.Module):
    """
    Mean-pooling over token dimension.
    Input: x of shape (B, L, D)
    Output: (B, D)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

class MaxPool(nn.Module):
    """
    Max-pooling over token dimension.
    Input: x of shape (B, L, D)
    Output: (B, D)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=1).values

class AttentionPool(nn.Module):
    """
    Attention-based pooling:
      - learns a global query vector
      - attends over tokens to produce a pooled representation.
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim,
                                          num_heads=num_heads,
                                          batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, D = x.shape
        q = self.query.expand(B, -1, -1)  # (B, 1, D)
        # MultiheadAttention expects (B, Seq, D)
        pooled, _ = self.attn(q, x, x)     # pooled: (B, 1, D)
        return pooled.squeeze(1)          # (B, D)

# Factory to select pooling by name
def get_pooling(name: str, dim: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name in ('mean', 'avg'):
        return MeanPool()
    elif name == 'max':
        return MaxPool()
    elif name in ('att', 'attention', 'attn'):
        return AttentionPool(dim=dim,
                             num_heads=kwargs.get('num_heads', 4))
    else:
        raise ValueError(f"Unknown pooling type '{name}'")
