# ──────────────────────────────────────────────────────────────
# scripts/mdn.py
# Utilities for a Mixture-Density Network (MDN) head.
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import (
    Normal,
    Categorical,
    MixtureSameFamily,
    Independent,          # ← NEW: wrap to treat last dim as event
)

# --------------------------------------------------------------------- head
class MDNHead(nn.Module):
    """
    Maps a feature vector h → parameters of a diagonal-covariance GMM.
    Outputs:
        π  : (B, K)        mixing probabilities
        μ  : (B, K, D)     component means
        σ  : (B, K, D)     component std-devs (positive)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_mixtures: int = 5,
        hidden: int | None = None,
    ):
        super().__init__()
        self.K = num_mixtures
        self.D = out_dim
        hid = hidden or 2 * in_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, (2 * out_dim + 1) * num_mixtures),
        )

    # --------------------------------------------------------- forward
    def forward(self, x: torch.Tensor):
        """
        Returns π (B,K), μ (B,K,D), σ (B,K,D)
        """
        b = x.size(0)
        raw = self.net(x).view(b, self.K, 1 + 2 * self.D)

        logit_pi   = raw[..., 0]                  # (B,K)
        mu         = raw[..., 1 : 1 + self.D]     # (B,K,D)
        log_sigma  = raw[..., 1 + self.D :]       # (B,K,D)

        pi    = F.softmax(logit_pi, dim=-1)
        sigma = F.softplus(log_sigma) + 1e-6      # avoid zero

        return pi, mu, sigma


# --------------------------------------------------------------------- loss
def mdn_nll(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor):
    """
    Negative log-likelihood of targets y under the predicted mixture.

    Args
    ----
    pi, mu, sigma : outputs from MDNHead
    y             : (B, D)  ground-truth vectors

    Returns
    -------
    Scalar mean NLL.
    """
    # Treat the final dimension (D) as the *event* dimension
    components = Independent(Normal(loc=mu, scale=sigma), 1)   # batch = (B,K)
    mixture    = Categorical(probs=pi)                         # batch = (B,)
    gmm        = MixtureSameFamily(mixture, components)

    nll = -gmm.log_prob(y.unsqueeze(1))    # (B,)  after broadcasting
    return nll.mean()
