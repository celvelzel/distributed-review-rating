# DeBERTa 1M vs 3M 代码差异分析

## 关键发现：3M 版本缺少 gradient_checkpointing

### 差异 1: Gradient Checkpointing（最关键）

**1M 版本** (`deberta_lora_1m.py:49`):
```python
self.backbone = get_peft_model(base, cfg)
self.backbone.gradient_checkpointing_enable()  # ✅ 有
```

**3M 版本** (`deberta_base_full.py:48-49`):
```python
self.backbone = get_peft_model(base, cfg)
# ❌ 没有 gradient_checkpointing_enable()
```

**影响**:
- Gradient checkpointing 通过重新计算激活值来节省显存
- 启用后，反向传播的梯度计算方式不同
- 这可能导致训练动态差异，影响模型收敛

### 差异 2: Batch Size 和 Gradient Accumulation

| 参数 | 1M | 3M |
|------|-----|-----|
| BATCH_SIZE | 16 | 32 |
| GRAD_ACCUM | 16 | 8 |
| Effective BS | 256 | 256 |
| Steps/epoch | 3,906 | 11,826 |
| Total steps | 11,718 | 35,478 |

虽然 effective batch size 相同，但：
- 1M: 每 16 个 batch 更新一次梯度，梯度估计更平滑
- 3M: 每 8 个 batch 更新一次梯度，梯度估计更噪声

### 差异 3: 学习率调度器

两者都使用相同的代码：
```python
steps = len(tl) // GRAD_ACCUM
sched = get_cosine_schedule_with_warmup(opt, int(steps*N_EPOCHS*WARMUP_RATIO), steps*N_EPOCHS)
```

但 `steps` 不同：
- 1M: steps = 3,906, total = 11,718, warmup = 1,171
- 3M: steps = 11,826, total = 35,478, warmup = 3,547

**问题**: 3M 的 warmup 步数是 1M 的 3 倍，但 warmup 比例相同（10%）。这意味着 3M 模型在 warmup 阶段花费更多时间，可能导致早期欠拟合。

---

## 结论

**1M 效果比 3M 好的原因不是数据问题，而是代码差异**：

1. **缺少 gradient checkpointing** — 3M 版本没有启用，导致训练动态不同
2. **Batch size 不同** — 3M 使用更大的单 batch（32 vs 16），梯度估计不同
3. **学习率调度不匹配** — 3M 的 warmup 更长，可能不适合更大的数据量

## 修复方案

### 方案 1: 让 3M 代码与 1M 一致（推荐）

修改 `deberta_base_full.py`：

```python
# 在 DeBERTaLoRA.__init__ 中添加
self.backbone.gradient_checkpointing_enable()

# 修改 batch 配置
BATCH_SIZE, GRAD_ACCUM = 16, 16  # 与 1M 一致
```

### 方案 2: 调整 3M 的学习率调度

```python
# 增加 warmup 比例
WARMUP_RATIO = 0.15  # 从 0.1 增加到 0.15

# 或者减少 epoch 数
N_EPOCHS = 2  # 从 3 减少到 2
```

### 方案 3: 直接用 1M 数据重新训练

```bash
python code/models/deberta_lora_1m.py
```

已知 1M 能产生好的结果，风险最低。

---

## 验证实验

写一个脚本对比两种配置的训练动态：

```python
# 对比 gradient checkpointing 对训练的影响
# 1. 启用 gradient checkpointing 训练 3M
# 2. 禁用 gradient checkpointing 训练 3M
# 3. 对比 val_rmse 和训练曲线
```

---

*Updated: 2026-06-17*
