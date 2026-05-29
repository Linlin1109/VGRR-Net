# ✈️ VGRR-Net: Visibility-Gap Re-indexing and Residual Reconstruction Network

> **A compact PyTorch reference implementation for visibility-gap-driven collaborative 3D perception in low-altitude flying-car scenes.**

This repository implements the core mechanism of **VGRR-Net**, a collaborative 3D perception framework that shifts multi-node collaboration from direct feature fusion to **gap-driven evidence supplementation**. The code is designed as a clean academic prototype that can be extended to real FCCP-3D-style datasets, multi-view image encoders, BEV lifting modules, and benchmark evaluation pipelines.

---

## 🌟 Method Intuition

Conventional collaborative perception pipelines often align and fuse all external features, which may introduce redundant, misaligned, or low-reliability information. VGRR-Net instead follows a more targeted route:

1. **Locate what the flying car has not observed well.**  
   The model estimates cell-level visibility and task importance to form a visibility-gap score.

2. **Treat collaboration as residual supplementation.**  
   External nodes provide only new evidence relative to the flying car's current 3D understanding.

3. **Weight evidence by communication reliability.**  
   Bandwidth, latency, and packet-loss states control how strongly each node contributes.

4. **Fill reliable residuals back into the ego representation.**  
   The final reconstructed 3D feature supports detection, semantic occupancy, and traversable-space prediction.

---

## 🧩 Repository Structure

```text
vgrrnet_code/
├── configs.py          # Dataclass-based model, training, and synthetic-data settings
├── dataset.py          # Synthetic FCCP-3D-style dataset and deterministic K-fold splitter
├── modules.py          # VGRR-Net modules: GRI, CRR, CRA, and multi-task heads
├── losses.py           # Multi-task objective, gap constraint, residual constraint, and metrics
├── train.py            # Reproducible training and validation loops
├── cross_validate.py   # Runnable K-fold cross-validation entry point
└── README.md           # Academic-style usage and verification notes
```

---

## ⚙️ Installation

The reference code only requires PyTorch and the Python standard library.

```bash
pip install torch
```

Recommended environment:

```text
Python >= 3.10
PyTorch >= 2.0
CPU or CUDA device
```

---

## 🚀 Quick Start

Run a lightweight three-fold validation with the synthetic FCCP-3D-style dataset:

```bash
python cross_validate.py --samples 36 --folds 3 --epochs 2 --batch-size 6 --device cpu
```

Optionally save the report:

```bash
python cross_validate.py --samples 36 --folds 3 --epochs 2 --batch-size 6 --output cv_report.json
```

---

## 🔬 Code-Level Correspondence to the Paper

| Paper concept | Code module | Role |
|---|---|---|
| Ego 3D feature representation | `SyntheticFCCP3D` input tensors | Provides cell-level 3D features for the flying-car ego node |
| Gap-aware re-indexing | `GapAwareReIndexing` | Estimates visibility, task importance, and visibility-gap scores |
| Communication residual reconstruction | `CommunicationResidualReconstruction` | Extracts residual evidence from aligned collaborative features |
| Communication reliability aggregation | `node_reliability` branch | Uses bandwidth, latency, and packet-loss states to weight nodes |
| Collaborative reconstructed representation | `reconstructed_features` | Fills reliable residuals into the gap-aware ego representation |
| Multi-task perception output | `MultiTaskPredictionHead` | Predicts detection classes, occupancy semantics, and traversable regions |
| Joint optimization | `compute_vgrr_loss` | Combines task loss, gap constraint, and residual constraint |

---

## 📊 Internal Three-Round Cross-Validation Refinement

The implementation was checked through three iterative validation rounds on a synthetic FCCP-3D-style dataset. These tests verify executable correctness, gradient flow, tensor compatibility, and basic learning behavior. They are not a substitute for full benchmark evaluation on the real FCCP-3D dataset.

| Round | Main update after validation | Mean validation loss | Det. acc. | Occ. acc. | Trav. IoU | Gap MAE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Initial GRI + CRR + CRA implementation | 3.4646 | 0.2384 | 0.1898 | 0.4562 | 0.1951 |
| 2 | Added gap-aware task head and class balancing | 3.4000 | 0.1968 | 0.2315 | 0.2865 | 0.1972 |
| 3 | Added spatial-aware task head and mild class balancing | **3.3728** | **0.2778** | 0.2245 | 0.2892 | 0.1967 |

A final extended verification using `36` samples, `3` folds, and `2` epochs reached:

```text
Mean validation loss : 3.0895
Detection accuracy  : 0.6412
Occupancy accuracy  : 0.2650
Traversable IoU     : 0.3407
Gap MAE             : 0.1861
```

---

## 🧠 Extending to Real FCCP-3D Data

To use this code with the full dataset described in the manuscript, replace `SyntheticFCCP3D` with a real data loader that returns the same keys:

```python
{
    "ego_features": Tensor[B, N, C],
    "agent_features": Tensor[B, A, N, C],
    "comm_state": Tensor[B, A, 3],
    "positions": Tensor[B, N, 3],
    "detection_labels": Tensor[B, N],
    "occupancy_labels": Tensor[B, N],
    "traversable_labels": Tensor[B, N, 1],
    "gap_target": Tensor[B, N, 1],
}
```

For a full paper-grade implementation, the following components should be connected before official experiments:

- a multi-view image encoder such as ResNet/FPN;
- a view-to-3D lifting module such as Lift-Splat-Shoot;
- geometric feature alignment from each collaborative node into the flying-car coordinate system;
- benchmark-specific 3D detection, occupancy, and traversability metrics;
- multi-seed training and significance testing.

---

## ✅ Design Notes

- The implementation keeps all major modules explicit rather than hiding the method inside a monolithic model class.
- All comments and docstrings are written in English for international paper/code release readiness.
- The synthetic dataset is deterministic, enabling reproducible smoke tests and regression checks.
- The code favors clarity and extensibility over maximal speed, which is appropriate for method-level academic release.

---

## 📌 Citation Placeholder

If this code is released with the manuscript, cite it as:

```bibtex
@article{vgrrnet2026,
  title   = {Visibility-Gap-Driven Collaborative 3D Perception for Flying Cars in Low-Altitude Urban Traffic},
  author  = {Anonymous Authors},
  journal = {Under Review},
  year    = {2026}
}
```
