"""Loss functions and lightweight metrics for VGRR-Net."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from configs import TrainConfig


def _balanced_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency weights for stable synthetic validation."""
    flat = labels.reshape(-1)
    counts = torch.bincount(flat, minlength=num_classes).float().to(labels.device)
    weights = torch.sqrt(counts.sum() / (counts.clamp_min(1.0) * float(num_classes)))
    weights = weights.clamp(0.5, 2.5)
    return weights / weights.mean().clamp_min(1e-6)


def compute_vgrr_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the multi-task, gap, and residual-constrained objective."""
    det_logits = outputs["detection_logits"].reshape(-1, outputs["detection_logits"].shape[-1])
    det_labels = batch["detection_labels"].reshape(-1)
    occ_logits = outputs["occupancy_logits"].reshape(-1, outputs["occupancy_logits"].shape[-1])
    occ_labels = batch["occupancy_labels"].reshape(-1)
    det_loss = F.cross_entropy(
        det_logits,
        det_labels,
        weight=_balanced_class_weights(det_labels, det_logits.shape[-1]),
    )
    occ_loss = F.cross_entropy(
        occ_logits,
        occ_labels,
        weight=_balanced_class_weights(occ_labels, occ_logits.shape[-1]),
    )
    trav_loss = F.binary_cross_entropy_with_logits(
        outputs["traversable_logits"], batch["traversable_labels"]
    )
    gap_loss = F.smooth_l1_loss(outputs["gap_score"], batch["gap_target"])

    non_gap_weight = 1.0 - batch["gap_target"]
    residual_energy = outputs["aggregated_residual"].pow(2).mean(dim=-1, keepdim=True)
    residual_loss = (non_gap_weight * residual_energy).mean()

    total = (
        cfg.lambda_det * det_loss
        + cfg.lambda_occ * occ_loss
        + cfg.lambda_trav * trav_loss
        + cfg.lambda_gap * gap_loss
        + cfg.lambda_residual * residual_loss
    )
    parts = {
        "loss": float(total.detach().cpu()),
        "det_loss": float(det_loss.detach().cpu()),
        "occ_loss": float(occ_loss.detach().cpu()),
        "trav_loss": float(trav_loss.detach().cpu()),
        "gap_loss": float(gap_loss.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
    }
    return total, parts


@torch.no_grad()
def compute_metrics(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute compact metrics for validation and cross-validation."""
    det_pred = outputs["detection_logits"].argmax(dim=-1)
    occ_pred = outputs["occupancy_logits"].argmax(dim=-1)
    trav_pred = (torch.sigmoid(outputs["traversable_logits"]) > 0.5).float()

    det_acc = (det_pred == batch["detection_labels"]).float().mean()
    occ_acc = (occ_pred == batch["occupancy_labels"]).float().mean()

    intersection = (trav_pred * batch["traversable_labels"]).sum()
    union = ((trav_pred + batch["traversable_labels"]) > 0).float().sum().clamp_min(1.0)
    trav_iou = intersection / union
    gap_mae = (outputs["gap_score"] - batch["gap_target"]).abs().mean()
    mean_reliability = outputs["node_reliability"].mean()

    return {
        "det_acc": float(det_acc.detach().cpu()),
        "occ_acc": float(occ_acc.detach().cpu()),
        "trav_iou": float(trav_iou.detach().cpu()),
        "gap_mae": float(gap_mae.detach().cpu()),
        "mean_reliability": float(mean_reliability.detach().cpu()),
    }
