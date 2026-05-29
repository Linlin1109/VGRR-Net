"""Synthetic FCCP-3D-style data for smoke tests and cross-validation.

The real FCCP-3D dataset described in the manuscript is not included here. This module
creates deterministic tensor samples that mimic ego 3D features, collaborative node
features, communication states, visibility-gap supervision, and task labels.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch.utils.data import Dataset, Subset

from configs import ModelConfig, SyntheticDataConfig


class SyntheticFCCP3D(Dataset):
    """A compact synthetic dataset for validating VGRR-Net code paths.

    Each item contains:
    - ego_features: flying-car ego 3D cell features, shape [N, C]
    - agent_features: collaborative-node features, shape [A, N, C]
    - comm_state: bandwidth, latency, and packet-loss values, shape [A, 3]
    - positions: normalized 3D cell coordinates, shape [N, 3]
    - detection_labels: cell-level detection labels, shape [N]
    - occupancy_labels: semantic occupancy labels, shape [N]
    - traversable_labels: binary traversable-space labels, shape [N]
    - gap_target: supervised visibility-gap scores, shape [N, 1]
    """

    def __init__(
        self,
        num_samples: int,
        model_cfg: ModelConfig,
        data_cfg: SyntheticDataConfig,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.model_cfg = model_cfg
        self.data_cfg = data_cfg
        self.generator = torch.Generator().manual_seed(seed)
        self.positions = self._build_positions(model_cfg.num_cells)
        self.samples = [self._make_sample(i) for i in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.samples[index]

    @staticmethod
    def _build_positions(num_cells: int) -> torch.Tensor:
        """Build normalized pseudo-3D coordinates for spatial cells."""
        side = int(round(num_cells ** (1.0 / 3.0)))
        side = max(side, 4)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(-1.0, 1.0, side),
                torch.linspace(-1.0, 1.0, side),
                torch.linspace(-0.2, 1.0, side),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 3)
        if coords.shape[0] < num_cells:
            repeat = num_cells - coords.shape[0]
            coords = torch.cat([coords, coords[:repeat]], dim=0)
        return coords[:num_cells]

    def _make_sample(self, index: int) -> Dict[str, torch.Tensor]:
        """Generate one deterministic sample with structured gaps and labels."""
        cfg = self.model_cfg
        data_cfg = self.data_cfg
        g = self.generator

        base = torch.randn(cfg.num_cells, cfg.feature_dim, generator=g) * 0.35
        spatial_signal = torch.sin(self.positions[:, :1] * 3.1) + torch.cos(self.positions[:, 1:2] * 2.3)
        semantic_axis = torch.linspace(-1.0, 1.0, cfg.feature_dim).unsqueeze(0)
        clean_features = base + 0.12 * spatial_signal * semantic_axis

        importance = torch.sigmoid(2.0 * spatial_signal + 0.3 * torch.randn(cfg.num_cells, 1, generator=g))
        visibility = torch.sigmoid(
            -1.5 * self.positions[:, 0:1]
            + 0.7 * self.positions[:, 2:3]
            + 0.4 * torch.randn(cfg.num_cells, 1, generator=g)
        )
        random_occ = torch.rand(cfg.num_cells, 1, generator=g) < data_cfg.occlusion_rate
        visibility = torch.where(random_occ, visibility * 0.45, visibility)
        gap_target = (1.0 - visibility) * importance

        ego_noise = torch.randn(cfg.num_cells, cfg.feature_dim, generator=g) * data_cfg.noise_std
        ego_features = clean_features * (0.55 + 0.45 * visibility) + ego_noise

        agent_features = []
        comm_states = []
        for agent_idx in range(cfg.num_agents):
            agent_offset = (agent_idx + 1) * 0.025
            agent_noise_std = torch.empty(1).uniform_(
                data_cfg.communication_noise[0],
                data_cfg.communication_noise[1],
                generator=g,
            ).item()
            coverage = torch.sigmoid(
                1.2 * self.positions[:, (agent_idx % 3) : (agent_idx % 3) + 1]
                + 0.5 * torch.randn(cfg.num_cells, 1, generator=g)
            )
            agent_view = clean_features * (0.65 + 0.35 * coverage) + agent_offset
            agent_view = agent_view + torch.randn(cfg.num_cells, cfg.feature_dim, generator=g) * agent_noise_std
            agent_features.append(agent_view)

            bandwidth = 1.0 - agent_noise_std
            latency = agent_noise_std * 2.0 + 0.05 * torch.rand(1, generator=g).item()
            packet_loss = torch.clamp(torch.tensor(agent_noise_std * 1.5), 0.0, 1.0).item()
            comm_states.append([bandwidth, latency, packet_loss])

        agent_features_tensor = torch.stack(agent_features, dim=0)
        comm_state_tensor = torch.tensor(comm_states, dtype=torch.float32)

        label_score = torch.sigmoid(clean_features[:, :1].mean(dim=-1, keepdim=True) + spatial_signal)
        foreground = label_score.squeeze(-1) > (1.0 - data_cfg.positive_ratio)
        detection_labels = torch.zeros(cfg.num_cells, dtype=torch.long)
        raw_classes = torch.bucketize(
            torch.sigmoid(clean_features[:, 1]),
            boundaries=torch.linspace(0.18, 0.82, cfg.num_detection_classes - 1),
        )
        detection_labels[foreground] = torch.clamp(raw_classes[foreground] + 1, 1, cfg.num_detection_classes - 1)

        occupancy_float = torch.sigmoid(
            1.1 * self.positions[:, 0] - 0.8 * self.positions[:, 1] + clean_features[:, 2] * 0.4
        )
        occupancy_labels = torch.bucketize(
            occupancy_float,
            boundaries=torch.linspace(0.2, 0.8, cfg.num_occupancy_classes - 1),
        ).long()
        traversable_labels = ((occupancy_labels == 0) | (occupancy_labels == 1)).float()

        return {
            "ego_features": ego_features.float(),
            "agent_features": agent_features_tensor.float(),
            "comm_state": comm_state_tensor.float(),
            "positions": self.positions.float(),
            "detection_labels": detection_labels,
            "occupancy_labels": occupancy_labels,
            "traversable_labels": traversable_labels.unsqueeze(-1),
            "gap_target": gap_target.float(),
        }


def make_kfold_subsets(dataset: Dataset, folds: int, seed: int = 42) -> List[Dict[str, Subset]]:
    """Create deterministic train/validation subsets without external dependencies."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    fold_size = max(1, len(indices) // folds)
    result: List[Dict[str, Subset]] = []
    for fold_idx in range(folds):
        start = fold_idx * fold_size
        end = len(indices) if fold_idx == folds - 1 else (fold_idx + 1) * fold_size
        valid_indices = indices[start:end]
        train_indices = indices[:start] + indices[end:]
        result.append({"train": Subset(dataset, train_indices), "valid": Subset(dataset, valid_indices)})
    return result
