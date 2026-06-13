# DeBERTa-v3-base LoRA 训练日志

**开始时间**: 2026-06-13 09:18 UTC
**状态**: 训练中 (Fold 1/5, Epoch 2)

---

## 训练配置

| 参数 | 值 |
|------|------|
| 模型 | microsoft/deberta-v3-base (86M params) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Target | query_proj, value_proj |
| 可训练参数 | 589,824 (0.32%) |
| Batch Size | 16 |
| Grad Accum | 16 (eff BS=256) |
| LR | 3e-5 |
| Epochs | 5 |
| Folds | 5 |
| Loss | CORAL Ordinal + R-Drop (α=0.5) |
| FP16 | True |

## 进度

| 时间 | 事件 | 备注 |
|------|------|------|
| 09:18 | 启动 Track B (LoRA) | 修复 allow_pickle + LoRA target modules |
| 09:18 | 跳过 Track A (全参数) | 速度太慢 (~110h), 仅运行 LoRA |
| 09:33 | Fold 1 Epoch 1 step 1000 | speed=121/s, ETA=19732s (~5.5h/epoch) |
| 14:30 | Fold 1 Epoch 2 step 18000 | speed=123/s, loss 0.6→0.3 (显著下降), ETA=17221s (~4.8h/epoch) |

## 时间预估 (更新于 14:30 UTC)

| 指标 | 值 |
|------|------|
| 当前速度 | 123 samples/s |
| 每 epoch | ~4.8h (含验证) |
| 每 fold (5 epochs) | ~24h |
| 总计 (5 folds) | ~120h |
| 已运行 | 6h 12m |
| 剩余 (Fold 1) | ~19h (epoch 2-5) |
| 30h 预算内 | 完成 Fold 1 + 部分 Fold 2 |
| **总计需要** | **~4 个 36h 作业** (checkpoint resume) |

### Checkpoint Resume 计划

| 作业 | 预计完成 | 累计进度 |
|------|---------|---------|
| Job 1 (当前) | Fold 1 完成 | 1/5 folds |
| Job 2 | Fold 2 完成 | 2/5 folds |
| Job 3 | Fold 3-4 完成 | 4/5 folds |
| Job 4 | Fold 5 + test prediction | 全部完成 |

### Loss 趋势

- Epoch 1: ~0.62 (初始)
- Epoch 2: ~0.33 (下降 47%)

## 问题修复

1. **allow_pickle 错误**: `np.load(cache_path)` 缺少 `allow_pickle=True`
2. **LoRA target 错误**: DeBERTa-v3 使用 `query_proj`/`value_proj` 而非 `query`/`value`

## Checkpoint 恢复

脚本自动保存 checkpoint → `artifacts/models/checkpoints_lora/fold{N}_epoch{N}.pt`
再次运行同一命令即可从断点继续:
```bash
python3.8 code/models/deberta_lora.py
```
