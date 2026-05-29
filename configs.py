"""Configuration objects for the VGRR-Net reference implementation."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Model hyperparameters used by the VGRR-Net modules."""

    num_cells: int = 24
    feature_dim: int = 32
    position_dim: int = 3
    hidden_dim: int = 64
    num_agents: int = 4
    num_detection_classes: int = 6  # five foreground categories plus background
    num_occupancy_classes: int = 5
    dropout: float = 0.15
    residual_scale: float = 0.50
    reliability_temperature: float = 1.25


@dataclass
class TrainConfig:
    """Training and validation settings."""

    seed: int = 42
    samples: int = 36
    folds: int = 3
    epochs: int = 2
    batch_size: int = 6
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    device: str = "cpu"
    lambda_det: float = 1.0
    lambda_occ: float = 0.8
    lambda_trav: float = 0.5
    lambda_gap: float = 0.35
    lambda_residual: float = 0.08


@dataclass
class SyntheticDataConfig:
    """Synthetic FCCP-3D-like tensor settings for runnable verification."""

    noise_std: float = 0.08
    occlusion_rate: float = 0.35
    communication_noise: Tuple[float, float] = (0.04, 0.18)
    positive_ratio: float = 0.28
