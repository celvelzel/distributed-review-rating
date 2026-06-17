# 3M Batch Size Ablation 实验计划

## 背景

3M DeBERTa (Kaggle 0.681) 远差于 1M (0.617)，差距 0.064。
已确认两个版本存在代码差异：
1. 缺少 gradient checkpointing（已修复）
2. Batch 配置不同：1M=16×16, 3M=32×8

本次实验只测 batch size 的影响，排除其他变量。

## 实验设计

### 假设

- **H1 (BS 噪声)**：3M 用 BS=32/GradAcc=8 导致梯度估计噪声更大 → 模型质量差
  - 预期：fold1 epoch1 Kaggle 明显优于现有 0.712
- **H2 (过拟合)**：3M 步数过多（7,813 steps/epoch vs 1M 的 3,906）导致过拟合
  - 预期：fold1 epoch1 Kaggle ≈ 0.712

### 变量控制

| 变量 | 本次实验 | 现有 3M | 1M |
|------|----------|---------|-----|
| 数据 | 3M (train_tokens.npz) | 3M | 1M |
| Model | deberta-v3-base + LoRA r=16 | 同 | 同 |
| BATCH_SIZE | **16** | 32 | 16 |
| GRAD_ACCUM | **16** | 8 | 16 |
| Effective BS | 256 | 256 | 256 |
| Gradient Checkpointing | **✅ 开启** | ❌ 关闭 | ✅ 开启 |
| R-Drop alpha | 0.5 | 0.5 | 0.5 |
| LR | 3e-5 | 3e-5 | 3e-5 |
| Folds × Epochs | **1f × 1e** | 3f × 3e | 3f × 3e |
| Steps/epoch | ~7,813 | ~7,813 | ~3,906 |

**唯一变量**：BATCH_SIZE + GRAD_ACCUM + gradient checkpointing

### 预期步骤数

```
3M 样本数 = 3,007,439
Fold1 训练集 = 3,007,439 × 2/3 ≈ 2,004,959
Steps/epoch = 2,004,959 / 16 (batch) / 16 (grad_accum) ≈ 7,832
```

（与现有 3M 的 7,813 steps/epoch 基本相同，因为 effective batch size 相同）

## 实施步骤

### Step 1: 创建实验脚本

基于 `deberta_lora_1m.py`，修改以下内容：

```python
# 文件: code/models/deberta_3m_bs16_ablation.py

# 改动 1: 数据路径
td = np.load(os.path.join(MODEL_DIR, "train_tokens.npz"), ...)   # 3M tokens
y_train = np.load(os.path.join(FEAT_DIR, "y_train.npy"))          # 3M labels

# 改动 2: 只训 1 fold, 1 epoch
N_FOLDS, N_EPOCHS = 1, 1

# 改动 3: 保持 1M 的 batch 配置
BATCH_SIZE, GRAD_ACCUM = 16, 16

# 改动 4: checkpoint 目录用新的
CKPT_DIR = os.path.join(MODEL_DIR, "checkpoints_3m_bs16_ablation")

# 改动 5: 输出文件名区分
np.save("deberta_3m_bs16_ablation_fold1e1_test.npy", ...)
```

### Step 2: 本地验证脚本可运行（不训练）

```bash
python code/models/deberta_3m_bs16_ablation.py --dry-run
```

确认：
- 数据加载成功
- 模型初始化成功
- 显存占用合理（应 < 8GB，因为有 gradient checkpointing）
- 步数计算正确

### Step 3: 提交 HPC 训练

```bash
# 预计时间: ~7,800 steps × 0.07s/step ≈ 9 分钟/epoch
# 但 3M 数据加载 + 验证 ≈ 总计 ~1 小时
```

### Step 4: 生成 Kaggle 提交

训练完成后，对 test set 预测并生成 CSV：
- `output/submission-3m-bs16-ablation-f1e1.csv`

### Step 5: 提交 Kaggle 并记录

```bash
kaggle competitions submit -c comp-5434-2526-sem-3-project \
  -f output/submission-3m-bs16-ablation-f1e1.csv \
  -m "3M BS16x16 ablation fold1 epoch1"
```

## 结果判读

| Kaggle RMSE | 结论 | 下一步 |
|-------------|------|--------|
| < 0.69 (明显优于 0.712) | H1 确认：BS 噪声是主因 | 用 BS=16 重训完整 3M |
| 0.69 - 0.72 (接近 0.712) | H2 确认：过拟合是主因 | 减少 epoch / 用 1M 数据 |
| > 0.72 (更差) | Gradient checkpointing 或其他因素 | 进一步诊断 |

## 风险

- **显存**：BS=16 + gradient checkpointing 在 12GB RTX 3080 Ti 上应无问题（1M 已验证）
- **时间**：单 fold 单 epoch 预计 ~1h，风险低
- **数据加载**：3M tokens 文件较大，加载需几分钟

## 文件清单

| 文件 | 用途 |
|------|------|
| `code/models/deberta_3m_bs16_ablation.py` | 实验脚本 |
| `artifacts/models/checkpoints_3m_bs16_ablation/` | checkpoint 目录 |
| `artifacts/models/deberta_3m_bs16_ablation_fold1e1_test.npy` | test 预测 |
| `output/submission-3m-bs16-ablation-f1e1.csv` | Kaggle 提交 |
