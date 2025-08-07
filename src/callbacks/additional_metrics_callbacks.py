# additional_metrics_callbacks.py
"""Lightning callback + helpers to compute extra evaluation metrics for
DLR-Hand-2 joint-angle regression.

Add this module to your PYTHONPATH and pass
    pl.Trainer(..., callbacks=[AdditionalMetrics(...)])
"""

from __future__ import annotations
import torch
import pytorch_lightning as pl
from typing import List, Dict, Tuple, Optional

# ───────────────────────────────────────────── helper functions ──

def best_of_k_rmse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 2:                        # k == 1
        return torch.mean((pred - gt).pow(2)).sqrt()
    per_h = (pred - gt.unsqueeze(1)).pow(2).mean(dim=-1)   # (B,k)
    return per_h.min(dim=1).values.mean().sqrt()

def top_logit_rmse(
    pred: torch.Tensor,
    logits: Optional[torch.Tensor],
    gt: torch.Tensor,
) -> torch.Tensor:
    if logits is None or pred.dim() == 2:
        return best_of_k_rmse(pred, gt)
    top    = logits.argmax(dim=1)                       # (B,)
    chosen = pred[torch.arange(pred.size(0)), top]      # (B,12)
    return torch.mean((chosen - gt).pow(2)).sqrt()

def pairwise_diversity(pred: torch.Tensor) -> torch.Tensor:
    if pred.dim() == 2:
        return torch.tensor(0.0, device=pred.device)
    B, k, _ = pred.shape
    diff = pred.unsqueeze(2) - pred.unsqueeze(1)        # (B,k,k,12)
    rms  = diff.pow(2).mean(dim=-1).sqrt()              # (B,k,k)
    mask = ~torch.eye(k, dtype=torch.bool, device=pred.device)
    return rms[:, mask].mean()

def coverage_at_tau(pred: torch.Tensor, gt: torch.Tensor, tau_deg: float = 10.0) -> torch.Tensor:
    tau_rad = torch.deg2rad(torch.tensor(tau_deg, device=pred.device))
    if pred.dim() == 2:
        return ((pred - gt).pow(2).mean(-1).sqrt() < tau_rad).float().mean()
    per_h = (pred - gt.unsqueeze(1)).pow(2).mean(-1).sqrt()
    return (per_h.min(dim=1).values < tau_rad).float().mean()

# ───────────────────────────────────────────── callback ─────────

class AdditionalMetrics(pl.Callback):
    """Computes and logs extra metrics for VAL and TEST epochs."""

    def __init__(self, tau_degrees: Tuple[float, float] = (10.0, 20.0)):
        super().__init__()
        self.tau_degrees = tau_degrees
        self._buffer: List[Dict[str, torch.Tensor]] = []
        self._buffer_test: List[Dict[str, torch.Tensor]] = []

    # ---------------- collect per-batch outputs ---------------

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._buffer.append({
            "pred_angles": outputs["pred_angles"].detach().float().cpu(),
            "pred_logits": (outputs.get("pred_logits").detach().float().cpu()
                            if outputs.get("pred_logits") is not None else None),
            "gt_angles":   outputs["gt_angles"].detach().float().cpu(),
        })

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._buffer_test.append({
            "pred_angles": outputs["pred_angles"].detach().float().cpu(),
            "pred_logits": (outputs.get("pred_logits").detach().float().cpu()
                            if outputs.get("pred_logits") is not None else None),
            "gt_angles":   outputs["gt_angles"].detach().float().cpu(),
        })

    # ---------------- aggregate & log -------------------------

    @staticmethod
    def _aggregate(buf: List[Dict[str, torch.Tensor]]):
        pred_list, logit_list, gt_list = [], [], []
        for d in buf:
            pred_list.append(d["pred_angles"])
            gt_list.append(d["gt_angles"])
            if d["pred_logits"] is not None:
                logit_list.append(d["pred_logits"])
        pred   = torch.cat(pred_list, 0)
        gt     = torch.cat(gt_list,   0)
        logits = torch.cat(logit_list, 0) if logit_list else None
        return pred, logits, gt

    def _log_metrics(self, trainer, prefix: str, pred, logits, gt):
        metrics = {
            f"{prefix}/best_k_RMSE":    best_of_k_rmse(pred, gt),
            f"{prefix}/top_logit_RMSE": top_logit_rmse(pred, logits, gt),
            f"{prefix}/pairwise_div":   pairwise_diversity(pred),
        }
        for tau in self.tau_degrees:
            metrics[f"{prefix}/coverage@{int(tau)}deg"] = coverage_at_tau(pred, gt, tau)
        for k, v in metrics.items():
            trainer.loggers[0].log_metrics({k: v.item()}, step=trainer.global_step)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._buffer:
            return
        pred, logits, gt = self._aggregate(self._buffer)
        self._log_metrics(trainer, "val", pred, logits, gt)
        self._buffer.clear()

    def on_test_epoch_end(self, trainer, pl_module):
        if not self._buffer_test:
            return
        pred, logits, gt = self._aggregate(self._buffer_test)
        self._log_metrics(trainer, "test", pred, logits, gt)
        self._buffer_test.clear()
