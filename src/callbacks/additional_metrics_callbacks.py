# additional_metrics_callbacks.py
"""Lightning callback + helpers to compute extra evaluation metrics for
DLR‑Hand‑2 joint‑angle regression.

Add this module to your PYTHONPATH and pass
    pl.Trainer(..., callbacks=[AdditionalMetrics(...)])

Assumptions
-----------
* The LightningModule's ``validation_step`` returns a dict with
  - ``pred_angles``   – shape (B,k,12) **or** (B,12) if k=1
  - ``pred_logits``   – shape (B,k) or None (optional)
  - ``gt_angles``     – shape (B,12)
  - ``gt_score``      – grasp quality scalar (B,) – optional

If you adopt the JointRegressor in your repo, simply modify its
``validation_step`` to return that dict (see README comment below).
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Dict, Tuple, Optional

# ───────────────────────────────────────────── helper functions ──

def best_of_k_rmse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """RMSE over the *closest* hypothesis per sample.

    pred: (B,k,12) or (B,12) – joint angles in **radians**
    gt:   (B,12)
    returns scalar Tensor
    """
    if pred.dim() == 2:  # k == 1
        err = pred - gt
        return torch.mean(err.pow(2)).sqrt()

    # (B,k,12) → (B,k)
    per_h = (pred - gt.unsqueeze(1)).pow(2).mean(dim=-1)
    best  = per_h.min(dim=1).values
    return best.mean().sqrt()


def top_logit_rmse(
    pred: torch.Tensor,
    logits: Optional[torch.Tensor],
    gt: torch.Tensor,
) -> torch.Tensor:
    """RMSE of the *arg‑max* hypothesis (the one the model thinks is best)."""
    if logits is None or pred.dim() == 2:
        return best_of_k_rmse(pred, gt)  # falls back to best‑of‑k
    top = logits.argmax(dim=1)           # (B,)
    chosen = pred[torch.arange(pred.size(0)), top]  # (B,12)
    err = chosen - gt
    return torch.mean(err.pow(2)).sqrt()


def pairwise_diversity(pred: torch.Tensor) -> torch.Tensor:
    """
    Average pairwise RMS distance between hypotheses.
    pred: (B,k,12)
    """
    if pred.dim() == 2:                         # k == 1 → no diversity
        return torch.tensor(0.0, device=pred.device)

    B, k, _ = pred.shape
    diff = pred.unsqueeze(2) - pred.unsqueeze(1)        # (B,k,k,12)
    rms  = diff.pow(2).mean(dim=-1).sqrt()              # (B,k,k)

    # build an off-diagonal mask and broadcast over batch
    mask = ~torch.eye(k, dtype=torch.bool, device=pred.device)  # (k,k)
    div  = rms[:, mask]                                         # (B,k*(k-1))
    return div.mean()                                           # scalar


def coverage_at_tau(pred: torch.Tensor, gt: torch.Tensor, tau_deg: float = 10.0) -> torch.Tensor:
    """Fraction of samples whose *closest* hypothesis lies within τ degrees."""
    tau_rad = torch.deg2rad(torch.tensor(tau_deg, device=pred.device))
    if pred.dim() == 2:
        per_s = (pred - gt).pow(2).mean(-1).sqrt()  # (B,)
        hit   = per_s < tau_rad
        return hit.float().mean()
    per_h = (pred - gt.unsqueeze(1)).pow(2).mean(-1).sqrt()  # (B,k)
    best  = per_h.min(dim=1).values
    return (best < tau_rad).float().mean()


# ───────────────────────────────────────────── callback ─────────

class AdditionalMetrics(pl.Callback):
    """Computes and logs extra metrics after each *validation* epoch."""

    def __init__(self, tau_degrees: Tuple[float, float] = (10.0, 20.0)):
        super().__init__()
        self.tau_degrees = tau_degrees
        self._buffer: List[Dict[str, torch.Tensor]] = []

    # ---------------- collect per‑batch outputs ---------------
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # ``outputs`` is whatever validation_step returned.
        self._buffer.append({
            "pred_angles": outputs["pred_angles"].detach().to(torch.float32).cpu(),
            "pred_logits": (outputs.get("pred_logits")
                            .detach().to(torch.float32).cpu()
                            if outputs.get("pred_logits") is not None else None),
            "gt_angles":   outputs["gt_angles"].detach().to(torch.float32).cpu(),
        })

    # ---------------- aggregate & log -------------------------
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._buffer:
            return

        pred_list, logit_list, gt_list = [], [], []
        for d in self._buffer:
            pred_list.append(d["pred_angles"])
            gt_list.append(d["gt_angles"])
            if d["pred_logits"] is not None:
                logit_list.append(d["pred_logits"].detach().cpu())

        pred = torch.cat(pred_list, dim=0)   # (N,k,12) or (N,12)
        gt   = torch.cat(gt_list,   dim=0)   # (N,12)
        logits = torch.cat(logit_list, dim=0) if logit_list else None

        metrics = {
            "val/best_k_RMSE":      best_of_k_rmse(pred, gt),
            "val/top_logit_RMSE":    top_logit_rmse(pred, logits, gt),
            "val/pairwise_div":      pairwise_diversity(pred),
        }
        for tau in self.tau_degrees:
            metrics[f"val/coverage@{int(tau)}deg"] = coverage_at_tau(pred, gt, tau)

        # log via Lightning
        for k, v in metrics.items():
            trainer.loggers[0].log_metrics({k: v.item()}, step=trainer.global_step)

        # clear buffer
        self._buffer.clear()
