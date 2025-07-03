# ──────────────────────────────────────────────────────────────
# scripts/mdn.py  
# ──────────────────────────────────────────────────────────────
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ head
class MDNHead(nn.Module):
    """
    Feature h  →  diagonal-covariance GMM parameters.

    Returns
    -------
    log_pi : (B,K)         log-mixing weights  (stable for log-sum-exp)
    mu     : (B,K,D)       means
    sigma  : (B,K,D)       std-devs  (σ ≥ var_min)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_mixtures: int = 5,
        hidden: int | None = None,
        var_min: float = 1e-3,           # ← variance floor
    ):
        super().__init__()
        self.K = num_mixtures
        self.D = out_dim
        self.var_min = var_min
        hid = hidden or 2 * in_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, (2 * out_dim + 1) * num_mixtures),
        )

    # ---------------------------------------------------------- forward
    def forward(self, x: torch.Tensor):
        b = x.size(0)
        raw = self.net(x).view(b, self.K, 1 + 2 * self.D)

        logit_pi  = raw[..., 0]                           # (B,K)
        mu        = raw[..., 1 : 1 + self.D]              # (B,K,D)
        log_sigma = raw[..., 1 + self.D :]                # (B,K,D)

        log_pi = F.log_softmax(logit_pi, dim=-1)          # stable logs
        sigma  = F.softplus(log_sigma) + self.var_min     # σ ≥ var_min
        return log_pi, mu, sigma


# ------------------------------------------------------------------ NLL
def mdn_nll(
    log_pi: torch.Tensor,        # (B,K)
    mu:     torch.Tensor,        # (B,K,D)
    sigma:  torch.Tensor,        # (B,K,D)
    y:      torch.Tensor,        # (B,D)
) -> torch.Tensor:
    """
    Exact negative log-likelihood of targets under the predicted mixture.
    Diagonal Gaussians, computed in log-space for stability.
    """
    y     = y.unsqueeze(1)                       # (B,1,D) → broadcast
    var   = sigma.pow(2)
    log_prob = -0.5 * (((y - mu) ** 2) / var + torch.log(2 * torch.pi * var)).sum(-1)  # (B,K)
    nll = -torch.logsumexp(log_pi + log_prob, dim=1)    # (B,)
    return nll.mean()
