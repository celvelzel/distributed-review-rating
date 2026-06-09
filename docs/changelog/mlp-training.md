# MLP v2: BERT-Only Training

## Architecture
- **Model**: 4-layer MLP with BatchNorm
- **Input**: DeBERTa (768d) only — LightGCN removed (near-zero embeddings)
- **Layers**: Linear(768→512) → BN → ReLU → Dropout(0.4)
             → Linear(512→256) → BN → ReLU → Dropout(0.4)
             → Linear(256→128) → BN → ReLU → Dropout(0.3)
             → Linear(128→1)
- **Loss**: MSE
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Scheduler**: CosineAnnealingLR

## Hyperparameters
- Folds: 5
- Epochs per fold: 50 (early stopping patience=10)
- Batch size: 4096
- Random seed: 42
- Device: cuda

## Results
- **OOF RMSE: 1.13119**
- **OOF pred std: 0.85802**
- CV time: 5947.6s
- Total time: 5996.4s

## Data
- Training samples: 3,007,439
- Test samples: 10,000
- Features: 768 (BERT-only, LightGCN removed)

## Outputs
- OOF predictions: `artifacts/models/mlp_oof.npy` (3,007,439,)
- Test predictions: `artifacts/models/mlp_test.npy` (10,000,)

## Changes from v1
- Removed LightGCN embeddings (near-zero, added noise)
- Reduced input dim: 896 → 768
- Added BatchNorm layers for training stability
- Increased dropout: 0.3 → 0.4
- Added cosine annealing LR scheduler
- Increased patience: 5 → 10
- Reduced batch size: 32768 → 2048
- Increased max epochs: 30 → 50

## Notes
- Predictions clipped to [1.0, 5.0]
- CPU fallback used if CUDA not available
