# DeBERTa 双线并行训练计划 — Training MiMo

> **状态**: Ready to Execute
>
> **目标**: 在 RTX 3080 Ti (12GB VRAM) 36h 作业限制内，完成 DeBERTa-v3-base 微调，获取最优 OOF 预测用于 Stacking 集成

---

## 概述

采用**双线并行**策略，同时运行两种微调方案，对比结果后选择更优的 Track 用于最终 Stacking。

| Track | 脚本 | 策略 | Folds × Epochs | 预估时间 |
|-------|------|------|----------------|---------|
| **A: 全参数微调** | `code/models/transformer_e2e.py` | Option C | 4f × 4e | ~10.4h |
| **B: LoRA 微调** | `code/models/deberta_lora.py` | LoRA r=16 | 5f × 5e | ~10h |

---

## 硬件约束

| 参数 | 值 |
|------|------|
| GPU | RTX 3080 Ti |
| VRAM | 12GB |
| 单次作业时限 | 36h |
| 可并行 GPU 任务 | 1 (建议串行或错峰) |

---

## Track A: 全参数微调 (Option C)

### 脚本
`code/models/transformer_e2e.py`

### 参数

| 参数 | 值 | 说明 |
|------|------|------|
| Model | microsoft/deberta-v3-base | 86M params |
| Pooling | Mean Pooling | 非 [CLS] |
| Loss | CORAL Ordinal + R-Drop (α=0.5) | 4 binary cumulative tasks |
| Batch Size | 12 | 显存安全 |
| Grad Accum | 21 | effective BS=252 |
| LR | 3e-5 | R-Drop 推荐 |
| Scheduler | Cosine (10% warmup) | |
| Epochs | 4 | Option C |
| Patience | 3 | |
| Folds | 4 | Option C |
| Max Length | 128 | 预缓存 token |
| FP16 | True | |
| Gradient Checkpointing | DISABLED | 显存够用 |

### 预估时间

| 阶段 | 时间 |
|------|------|
| 数据加载 + tokenization | ~5min (有缓存) |
| Fold 1 (4 epochs) | ~2.6h |
| Fold 2-4 | ~7.8h |
| Test prediction | ~10min |
| **总计** | **~10.4h** |

### Checkpoint 行为

- 每 epoch 保存一次 → `artifacts/models/checkpoints/fold{N}_epoch{N}.pt`
- 自动跳过已完成 fold
- 成功完成后自动清理 checkpoint

---

## Track B: LoRA 微调

### 脚本
`code/models/deberta_lora.py`

### 参数

| 参数 | 值 | 说明 |
|------|------|------|
| Model | microsoft/deberta-v3-base | 86M params (冻结) |
| LoRA Rank | 16 | |
| LoRA Alpha | 32 | |
| LoRA Target | query, value | Attention 层 |
| LoRA Dropout | 0.05 | |
| Pooling | Mean Pooling | |
| Loss | CORAL Ordinal + R-Drop (α=0.5) | |
| Batch Size | 16 | LoRA 显存更低 |
| Grad Accum | 16 | effective BS=256 |
| LR | 3e-5 | |
| Scheduler | Cosine (10% warmup) | |
| Epochs | 5 | |
| Patience | 3 | |
| Folds | 5 | |
| Max Length | 128 | |
| FP16 | True | |
| Gradient Checkpointing | DISABLED | LoRA 显存够用 |

### 预估时间

| 阶段 | 时间 |
|------|------|
| 数据加载 + tokenization | ~5min (有缓存) |
| Fold 1 (5 epochs) | ~2h |
| Fold 2-5 | ~8h |
| Test prediction | ~10min |
| **总计** | **~10h** |

### Checkpoint 行为

- 每 epoch 保存一次 → `artifacts/models/checkpoints_lora/fold{N}_epoch{N}.pt`
- 自动跳过已完成 fold
- 成功完成后自动清理 checkpoint

---

## 执行顺序

### 推荐方案: 先跑 Track B (LoRA)

理由:
1. LoRA 更快完成 (10h vs 10.4h)
2. LoRA 显存更低 (2-3GB vs 4.4GB)，不会与其他任务冲突
3. 如果 LoRA 效果足够好，可以省下时间跑其他任务

```bash
# Step 1: 先跑 Track B
python code/models/deberta_lora.py

# Step 2: 如果有时间，再跑 Track A
python code/models/transformer_e2e.py
```

### 备选方案: 串行跑两个 Track

如果 36h 内能跑完两个:
```bash
# 先 Track B，再 Track A
python code/models/deberta_lora.py && python code/models/transformer_e2e.py
```

---

## 结果对比

训练完成后，对比两个 Track 的 OOF RMSE:

| 指标 | Track A (全参数) | Track B (LoRA) | 胜出 |
|------|-----------------|----------------|------|
| OOF RMSE | TBD | TBD | ? |
| 训练时间 | ~10.4h | ~10h | |
| 显存峰值 | ~4.4GB | ~2-3GB | |

选择 OOF RMSE 更低的 Track 用于 Stacking 集成。

---

## 输出文件

### Track A 输出
```
artifacts/models/deberta_e2e_oof.npy    (3,007,439,)
artifacts/models/deberta_e2e_test.npy   (10,000,)
artifacts/models/deberta_e2e_fold{1-4}.pt
```

### Track B 输出
```
artifacts/models/deberta_lora_oof.npy   (3,007,439,)
artifacts/models/deberta_lora_test.npy  (10,000,)
artifacts/models/deberta_lora_fold{1-5}.pt
```

### Checkpoint 目录
```
artifacts/models/checkpoints/      ← Track A
artifacts/models/checkpoints_lora/ ← Track B
```

---

## HPC 作业脚本参考

```bash
#!/bin/bash
#SBATCH --job-name=deberta-lora
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=36:00:00

cd /path/to/distributed-review-rating

# 激活环境
source activate your_env

# 运行 Track B (LoRA)
python code/models/deberta_lora.py

# 如果 Track B 完成且有时间，运行 Track A
# python code/models/transformer_e2e.py
```

---

## Checkpoint 恢复说明

如果 36h 内未完成，再次提交同样的作业命令即可:

```bash
# 脚本会自动检测 checkpoint 并从断点继续
python code/models/deberta_lora.py
```

输出示例:
```
[CKPT] Resuming from: fold3_epoch2.pt
[CKPT] Resumed fold 3 from epoch 2, continuing from epoch 3
[CKPT] best_val_rmse=0.82345, patience=1
```

---

## 验证检查清单

- [ ] Track B (LoRA) 训练完成
- [ ] Track B OOF RMSE 记录
- [ ] Track A 训练完成 (如有时间)
- [ ] Track A OOF RMSE 记录
- [ ] 对比两个 Track 结果
- [ ] 选择更优 Track 用于 Stacking
- [ ] Test predictions 生成

---

## 文件清单

### 训练脚本
- `code/models/transformer_e2e.py` — Track A: 全参数微调 (Option C: 4f × 4e)
- `code/models/deberta_lora.py` — Track B: LoRA 微调 (5f × 5e)

### 参考计划
- `.sisyphus/plans/kaggle-optimization-qwen.md` — 原始优化计划
- `.sisyphus/plans/optimization-chatglm.md` — ChatGLM 方案参考
- `.sisyphus/plans/optimization-gemini.md` — Gemini 方案参考 (LoRA 来源)

### 进度报告
- `docs/progress/kaggle-optimization-progress.md` — 整体进度

---

*Plan created by MiMo on 2026-06-12*
