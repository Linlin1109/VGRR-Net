"""Three-fold cross-validation entry point for the VGRR-Net reference code.

This script uses the synthetic FCCP-3D-style dataset to verify that all modules,
losses, gradients, and metrics are executable without access to private data.
"""

from __future__ import annotations

import argparse
import json
from statistics import mean, pstdev
from typing import Dict, List

from configs import ModelConfig, SyntheticDataConfig, TrainConfig
from dataset import SyntheticFCCP3D, make_kfold_subsets
from train import fit_fold, seed_everything


def summarize(fold_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Summarize fold-level results with mean and population standard deviation."""
    keys = sorted(fold_results[0].keys())
    summary = {}
    for key in keys:
        values = [item[key] for item in fold_results]
        summary[key] = {"mean": mean(values), "std": pstdev(values)}
    return summary


def run_cross_validation(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    data_cfg: SyntheticDataConfig,
) -> Dict[str, object]:
    """Run deterministic k-fold validation and return a serializable report."""
    seed_everything(train_cfg.seed)
    dataset = SyntheticFCCP3D(
        num_samples=train_cfg.samples,
        model_cfg=model_cfg,
        data_cfg=data_cfg,
        seed=train_cfg.seed,
    )
    folds = make_kfold_subsets(dataset, train_cfg.folds, seed=train_cfg.seed)
    fold_results = []
    for fold_idx, split in enumerate(folds):
        result = fit_fold(
            split["train"],
            split["valid"],
            model_cfg,
            train_cfg,
            fold_seed=train_cfg.seed + fold_idx,
        )
        fold_results.append(result)
        print(f"Fold {fold_idx + 1}/{train_cfg.folds}: {json.dumps(result, indent=2)}")
    report = {"folds": fold_results, "summary": summarize(fold_results)}
    print("Summary:")
    print(json.dumps(report["summary"], indent=2))
    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run VGRR-Net synthetic cross-validation.")
    parser.add_argument("--samples", type=int, default=96, help="Number of synthetic samples.")
    parser.add_argument("--folds", type=int, default=3, help="Number of validation folds.")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--device", type=str, default="cpu", help="Device identifier.")
    parser.add_argument("--output", type=str, default="", help="Optional JSON report path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        samples=args.samples,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    data_cfg = SyntheticDataConfig()
    cv_report = run_cross_validation(model_cfg, train_cfg, data_cfg)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(cv_report, f, indent=2)
