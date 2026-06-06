# T19: MLP Base Model Training

## Architecture
- **Model**: 3-layer MLP (RatingMLP)
- **Input**: DeBERTa (768d) + LightGCN user_emb (64d) + LightGCN item_emb (64d) = **896d**
- **Layers**: Linear(896→512) → ReLU → Dropout(0.3) → Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→1)
- **Loss**: MSE
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)

## Hyperparameters
- Folds: 5 (stratified-shuffle split, seed=42)
- Epochs per fold: 30 (early stopping patience=5)
- Batch size: 32768
- Device: NVIDIA GeForce RTX 3080 Ti (CUDA)

## Per-Fold Results

| Fold | Epochs | Best Val RMSE | Early Stop? |
|------|--------|---------------|-------------|
| 1    | 30     | 1.14625       | No          |
| 2    | 30     | 1.14830       | No          |
| 3    | 18     | 1.15448       | Yes (epoch 18) |
| 4    | 30     | 1.14838       | No          |
| 5    | 17     | 1.16261       | Yes (epoch 17) |

### Training Loss Curves (selected epochs)

**Fold 1:**
| Epoch | Train Loss | Val Loss  | Val RMSE  |
|-------|------------|-----------|-----------|
| 1     | 2.73498    | 1.57397   | 1.25458   |
| 5     | 1.53608    | 1.36702   | 1.16920   |
| 10    | 1.50606    | 1.35537   | 1.16420   |
| 15    | 1.48483    | 1.33812   | 1.15677   |
| 20    | 1.46850    | 1.32648   | 1.15173   |
| 25    | 1.45391    | 1.31921   | 1.14857   |
| 30    | 1.43528    | 1.31390   | 1.14625   |

**Fold 3 (early stopped):**
| Epoch | Train Loss | Val Loss  | Val RMSE  |
|-------|------------|-----------|-----------|
| 1     | 2.69748    | 1.59142   | 1.26152   |
| 5     | 1.53321    | 1.37833   | 1.17402   |
| 10    | 1.49097    | 1.35304   | 1.16320   |
| 15    | 1.46041    | 1.36883   | 1.16997   |
| 18    | —          | —         | early stop|

## Summary
- **OOF RMSE: 1.15201**
- CV training time: 3,957s (~66 min)
- Total time: 4,132s (~69 min)

## Data
- Training samples: 3,007,439
- Test samples: 10,000
- Features: 896 (768 DeBERTa + 64 user + 64 item)
- User/item embedding coverage: ~100% (zero-padded for unknowns)

## Outputs
- OOF predictions: `artifacts/models/mlp_oof.npy` shape=(3,007,439,) range=[1.0, 5.0] mean=3.916
- Test predictions: `artifacts/models/mlp_test.npy` shape=(10,000,) range=[1.21, 4.96] mean=3.966

## Notes
- Users/items not found in LightGCN embeddings are zero-padded
- Predictions clipped to [1.0, 5.0]
- GPU: NVIDIA GeForce RTX 3080 Ti
- The MLP RMSE (1.152) is higher than the LightGBM stage-2 model (~0.55) as expected — the MLP uses only embeddings, not the full 5927-feature set. This model is intended as a base model for stacking, providing decorrelated predictions.
