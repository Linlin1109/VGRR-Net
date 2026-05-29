"""Training and validation loops for the VGRR-Net reference code."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset

from configs import ModelConfig, TrainConfig
from losses import compute_metrics, compute_vgrr_loss
from modules import VGRRNet


def seed_everything(seed: int) -> None:
    """Set deterministic random seeds for reproducible verification."""
    random.seed(seed)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move a dictionary batch to the requested device."""
    return {key: value.to(device) for key, value in batch.items()}


def average_dict(values: Dict[str, list]) -> Dict[str, float]:
    """Average a dictionary of scalar lists."""
    return {key: float(sum(items) / max(1, len(items))) for key, items in values.items()}


def train_one_epoch(
    model: VGRRNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch."""
    model.train()
    tracker = defaultdict(list)
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            batch["ego_features"],
            batch["agent_features"],
            batch["comm_state"],
            batch["positions"],
        )
        loss, loss_parts = compute_vgrr_loss(outputs, batch, train_cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
        optimizer.step()
        for key, value in loss_parts.items():
            tracker[key].append(value)
    return average_dict(tracker)


@torch.no_grad()
def validate(
    model: VGRRNet,
    loader: DataLoader,
    train_cfg: TrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate the model on a validation split."""
    model.eval()
    tracker = defaultdict(list)
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(
            batch["ego_features"],
            batch["agent_features"],
            batch["comm_state"],
            batch["positions"],
        )
        _, loss_parts = compute_vgrr_loss(outputs, batch, train_cfg)
        metrics = compute_metrics(outputs, batch)
        for key, value in {**loss_parts, **metrics}.items():
            tracker[key].append(value)
    return average_dict(tracker)


def fit_fold(
    train_set: Dataset,
    valid_set: Dataset,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    fold_seed: int,
) -> Dict[str, float]:
    """Train and validate one cross-validation fold."""
    seed_everything(fold_seed)
    device = torch.device(train_cfg.device)
    model = VGRRNet(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    last_train = {}
    last_valid = {}
    for _ in range(train_cfg.epochs):
        last_train = train_one_epoch(model, train_loader, optimizer, train_cfg, device)
        last_valid = validate(model, valid_loader, train_cfg, device)

    return {f"train_{k}": v for k, v in last_train.items()} | {f"valid_{k}": v for k, v in last_valid.items()}
