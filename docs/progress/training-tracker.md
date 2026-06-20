# COMP5434 Training Progress Tracker
_Last updated: 2026-06-20 12:00_

## Current Status

| Component | Status | Details |
|-----------|--------|---------|
| **v3-base** | ✅ COMPLETE | 3f×3e, all 9 checkpoints saved |
| **v3-large** | 🔄 RUNNING | LR=2e-5, Fold1 Ep3 进行中 |
| **3M 5f×5e** | ⏸️ STOPPED | LoRA 过拟合确认 (val_rmse 1.396) |
| **Stacking V3** | ✅ COMPLETE | OOF=1.118, ablation done |
| **Kaggle Best** | **0.61473** | VE 88% + Stacking V3 ridge+lgb 12% |
| **Target** | < 0.47361 | Beat 2nd place (Deepsick) |

---

## Base Model (DeBERTa-v3-base, 86M) — ✅ COMPLETE

| Fold | Epoch 1 | Epoch 2 | Epoch 3 |
|------|---------|---------|---------|
| 1 | 1.14668 | 1.13945 | 1.13857 |
| 2 | 1.14635 | 1.13915 | **1.13551** ✅ |
| 3 | 1.14563 | 1.13876 | 1.13693 |

- Config: LoRA r=16 α=32, BS=32, GradAcc=8, LR=3e-5, CORAL+R-Drop
- Training time: ~12h total
- All checkpoints: `artifacts/models/checkpoints_base_full/`
- Fold 2 e3 补训完成 (GPU swap strategy)

---

## Large Model (DeBERTa-v3-large, 435M) — 🔄 RUNNING

| Fold | Epoch 1 | Epoch 2 | Epoch 3 |
|------|---------|---------|---------|
| 1 | ~1.1787 | **1.15961** | 🔄 进行中 |
| 2 | — | — | — |
| 3 | — | — | — |

- Config: LoRA r=16 α=32, BS=16, GradAcc=16, **LR=2e-5**, CORAL+R-Drop
- PID: 1821809
- Checkpoints: `artifacts/models/checkpoints_large_full/` (fold1_e1, fold1_e2)
- **Note:** 从 fold1_epoch2 恢复，LR 从 1e-5 提高到 2e-5
- ETA: ~36h (2026-06-21 ~00:00)

---

## Performance Comparison (Epoch 2)

| Model | Params | LR | val_rmse e2 |
|-------|--------|-----|------------|
| v3-base | 86M | 3e-5 | **1.139** |
| v3-large | 435M | 1e-5 | 1.160 |

Large model is **1.9% worse** at epoch 2 despite 3.5x more params.

---

## Resource Usage

| Resource | Base Training | Large Training |
|----------|--------------|----------------|
| GPU VRAM | 7.7GB (61%) | 10.5GB (83%) |
| System RAM | 10.9GB | 12.1GB |
| Speed | ~100 steps/s | ~1.1 steps/s |
| Time/epoch | ~1.7h | ~10.3h |
| Total time | ~12h | ~93h (est) |

---

## Kaggle Submissions (Best 10)

| Rank | Score | Submission | Date |
|------|-------|-----------|------|
| 1 | **0.61473** | sub-deb1m-ve88-sv3rlg12 | Jun 20 |
| 2 | 0.61725 | submission-deb1m-ve90-sv3-10 | Jun 18 |
| 3 | 0.61734 | dve90-r10 | Jun 15 |
| 4 | 0.61733 | sub-deb1m-ve90-sv3lgb10 | Jun 20 |
| 5 | 0.62029 | sub-deb1m-ve92-sv3r8 | Jun 20 |
| 6 | 0.62463 | dve95-r5 | Jun 15 |
| 7 | 0.62468 | sub-deb1m85-basemulti10-sv3rlg5 | Jun 20 |
| 8 | 0.63287 | deberta-ve | Jun 15 |
| 9 | 0.63449 | base_ve_90_small_ve_10 | Jun 16 |
| 10 | 0.66376 | stacking-v2 | Jun 14 |

## Ablation Study Results (2026-06-18)

| ID | Kaggle RMSE | vs Baseline | 结论 |
|----|-------------|-------------|------|
| A1 (v2-ridge) | **0.61734** | — | baseline |
| A2 (v3-ridge) | 0.61746 | +0.00012 | Graph features 无效 |
| A3 (v3-ridge-lgb) | **0.61725** | -0.00009 | Ridge+LGB 边际改进 |
| A4 (3M BS16) | 0.74265 | +0.12531 | **H2 确认：3M 过拟合** |

**关键发现**: 
1. Stacking V3 改进是边际的 (±0.0002)。瓶颈是 DeBERTa 模型质量。
2. 3M 模型过拟合严重，BS=16 配置更差 (0.74265 vs 0.68126)。
3. 1M 模型是最佳选择，3M 路径应放弃。

---

## Key Findings

1. **旧预测 (deberta_lora_fold1_test.npy) 精确匹配 0.61734** — 新 full 3M 训练的预测 Kaggle=0.69，远差于旧预测。需要调查原因。
2. **Small model 无法超越 base model** — 8 次提交确认。
3. **Cross-fold 平均需要 OOF 对齐** — 直接平均不同 fold 的预测有害 (0.71 vs 0.63)。
4. **LR 对 large model 至关重要** — 1e-5 可能太低，建议提高到 2e-5。
5. **Graph features 对 Kaggle 无效** — 消融实验证明，可移除以简化 pipeline。
6. **Meta-learner 升级无效** — Ridge 已接近最优，Ridge+LGB 只有 -0.00009 改进。
7. **瓶颈在 DeBERTa 模型质量** — 需要更好的基础模型，不是 meta-learner 复杂度。
8. **3M OOF 保存有 bug** — resume 逻辑导致只有 1/3 非零。

---

## Next Steps

### 短期 (立即)
1. **更新 Kaggle token** — 当前 token 过期
2. **等 3M BS16 消融完成** — 验证 batch size 影响
3. **修复 3M OOF bug** — 从 checkpoint 重新生成

### 中期 (1-3 天)
1. **DeBERTa-v3-large** — 提高 LR 到 2e-5，重新训练
2. **多折集成** — 用 3 折预测的加权平均
3. **伪标签** — 用高置信度预测扩充训练集

### 长期 (3-7 天)
1. **异构集成** — DeBERTa-base + DeBERTa-large + XGBoost
2. **新特征探索** — 句法结构、情感强度、评论长度
3. **目标: 超过第2名 (Kaggle < 0.47361)**

---

## Process IDs

| Process | PID | Status |
|---------|-----|--------|
| v3-base training | 1266949 | ✅ Done |
| v3-large training | 1380155 | ⏸️ Paused (killed after e2 checkpoint) |
| fold2 e3 retrain (GPU) | 1366029 | ✅ Done |
| Auto-launch monitor | 1291938 | — |

---

## File Locations

| Purpose | Path |
|---------|------|
| Base checkpoints | `artifacts/models/checkpoints_base_full/` |
| Large checkpoints | `artifacts/models/checkpoints_large_full/` |
| Base training log | `artifacts/base_full_training.log` |
| Large training log | `artifacts/large_full_training.log` |
| Fold2 e3 GPU log | `artifacts/fold2_e3_gpu.log` |
| Old predictions (best) | `artifacts/models/deberta_lora_fold1_test.npy` |
| Dashboard | `tech_dashboard.html` |
| Optimization plan | `docs/plans/post-base-optimization.md` |
