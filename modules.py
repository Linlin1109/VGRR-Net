"""Core VGRR-Net modules.

The implementation focuses on the algorithmic mechanism proposed in the manuscript:
visibility-gap modeling, gap-aware re-indexing, communication residual extraction,
reliability-weighted aggregation, and multi-task prediction.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from configs import ModelConfig


class MLP(nn.Module):
    """A small normalized MLP block used throughout VGRR-Net."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GapAwareReIndexing(nn.Module):
    """Estimate visibility, task importance, and gap-aware ego features."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_dim = cfg.feature_dim + cfg.position_dim
        self.visibility_head = MLP(in_dim, cfg.hidden_dim, 1, cfg.dropout)
        self.importance_head = MLP(in_dim, cfg.hidden_dim, 1, cfg.dropout)
        self.reindex_proj = MLP(cfg.feature_dim, cfg.hidden_dim, cfg.feature_dim, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.feature_dim)

    def forward(self, ego_features: torch.Tensor, positions: torch.Tensor) -> Dict[str, torch.Tensor]:
        descriptor = torch.cat([ego_features, positions], dim=-1)
        visibility = torch.sigmoid(self.visibility_head(descriptor))
        importance = torch.sigmoid(self.importance_head(descriptor))
        gap_score = (1.0 - visibility) * importance
        delta = self.reindex_proj(ego_features)
        reindexed = self.norm(ego_features + gap_score * delta)
        return {
            "visibility": visibility,
            "importance": importance,
            "gap_score": gap_score,
            "reindexed_features": reindexed,
        }


class CommunicationResidualReconstruction(nn.Module):
    """Extract and aggregate collaborative residuals around visibility gaps."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.agent_proj = MLP(cfg.feature_dim, cfg.hidden_dim, cfg.feature_dim, cfg.dropout)
        self.ego_proj = MLP(cfg.feature_dim, cfg.hidden_dim, cfg.feature_dim, cfg.dropout)
        self.reliability_head = MLP(3, cfg.hidden_dim // 2, 1, cfg.dropout)
        self.fusion_proj = MLP(cfg.feature_dim, cfg.hidden_dim, cfg.feature_dim, cfg.dropout)
        self.norm = nn.LayerNorm(cfg.feature_dim)

    def forward(
        self,
        reindexed_features: torch.Tensor,
        agent_features: torch.Tensor,
        comm_state: torch.Tensor,
        gap_score: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ego_term = self.ego_proj(reindexed_features).unsqueeze(1)
        agent_term = self.agent_proj(agent_features)
        raw_residual = gap_score.unsqueeze(1) * (agent_term - ego_term)

        reliability_logits = self.reliability_head(comm_state) / self.cfg.reliability_temperature
        reliability = torch.sigmoid(reliability_logits).unsqueeze(2)
        weighted_residual = raw_residual * reliability
        normalizer = reliability.sum(dim=1).clamp_min(1e-6)
        aggregated = weighted_residual.sum(dim=1) / normalizer

        reconstructed = self.norm(
            reindexed_features + self.cfg.residual_scale * self.fusion_proj(aggregated)
        )
        return {
            "node_reliability": reliability.squeeze(2),
            "node_residual": raw_residual,
            "aggregated_residual": aggregated,
            "reconstructed_features": reconstructed,
        }


class MultiTaskPredictionHead(nn.Module):
    """Prediction head for detection, occupancy, and traversable-space estimation.

    The head receives reconstructed 3D features, the learned gap score, and
    normalized spatial coordinates. This preserves spatial context for 3D cell
    classification while keeping the gap-aware decision path explicit.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(cfg.feature_dim + 1 + cfg.position_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.det_head = nn.Linear(cfg.hidden_dim, cfg.num_detection_classes)
        self.occ_head = nn.Linear(cfg.hidden_dim, cfg.num_occupancy_classes)
        self.trav_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        gap_score: torch.Tensor,
        positions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        head_input = torch.cat([features, gap_score, positions], dim=-1)
        shared = self.shared(head_input)
        return {
            "detection_logits": self.det_head(shared),
            "occupancy_logits": self.occ_head(shared),
            "traversable_logits": self.trav_head(shared),
        }


class VGRRNet(nn.Module):
    """Visibility-Gap Re-indexing and Residual Reconstruction Network."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_norm = nn.LayerNorm(cfg.feature_dim)
        self.gap_reindexing = GapAwareReIndexing(cfg)
        self.residual_reconstruction = CommunicationResidualReconstruction(cfg)
        self.head = MultiTaskPredictionHead(cfg)

    def forward(
        self,
        ego_features: torch.Tensor,
        agent_features: torch.Tensor,
        comm_state: torch.Tensor,
        positions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ego_features = self.input_norm(ego_features)
        agent_features = self.input_norm(agent_features)
        gap_outputs = self.gap_reindexing(ego_features, positions)
        recon_outputs = self.residual_reconstruction(
            gap_outputs["reindexed_features"],
            agent_features,
            comm_state,
            gap_outputs["gap_score"],
        )
        predictions = self.head(
            recon_outputs["reconstructed_features"],
            gap_outputs["gap_score"],
            positions,
        )
        return {**gap_outputs, **recon_outputs, **predictions}
