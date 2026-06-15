# Phase 5 Clinical Benchmark Report: MicroUNet
Date: 2026-06-06
MLflow Run ID: 7c1248c28c23412e9689dff4f2151b74

## 🏗️ Model Architecture
- Type: MONAI BasicUNet
- Features: (8, 8, 16, 16, 32, 8) [Nano-Micro Configuration]
- Activation: ReLU
- Normalization: GroupNorm (4 groups)
- Initialization: Kaiming Normal (relu-optimized)

## ⚙️ Hyperparameters
- Optimizer: SGD (Momentum=0.0)
- Learning Rate: 5e-7
- Logical Batch Size: 4 (Accumulated)
- Physical Batch Size: 1
- Max Grad Norm: 1.0 (Opacus Clipping)
- Privacy Budget: Target Eps=10.0, Delta=1e-5

## 📊 Performance Metrics
| Center | Hospital | Samples | Round 1 Loss | Round 2 Loss |
| :--- | :--- | :---: | :---: | :---: |
| 0 | Guy's | 238 | 0.0081 | 0.0071 |
| 1 | Hammersmith | 145 | 0.4689 | 0.4394 |
| 2 | IOP | 59 | 0.5125 | 0.5228 |
| **Global** | | **442** | **0.330** | **0.323** |

## 🛡️ Privacy & Stability
- Final Cumulative Epsilon: 8.12
- Weight Sanitizer: Active (NaN/Inf replaced with 0.0, clipped to [-10, 10])
- Data Scrubbing: Center 0 removed 11 samples with fragmented masks.

## 📡 Telemetry Fix
- Multi-Process logging implemented via `MlflowClient`. 
- Workers receive `run_id` from orchestrator and log directly to parent run.
- Environment: PyTorch 2.4.1, NumPy 1.26.4.
