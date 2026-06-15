# Kaggle Optimization V3 — Push Towards 0.5

## TL;DR

> **Quick Summary**: 突破当前 0.617 瓶颈，通过多折集成、OOF 校准、更大模型和全量数据训练，目标 RMSE → 0.55-0.58
> 
> **Current Best**: 0.61734 (DeBERTa-v3-base VE 90% + Ridge 10%)
> **Target**: 0.5
> **Gap**: 19%

---

## 资源瓶颈分析

### 当前资源限制

| 资源 | 当前限制 | 瓶颈程度 | 说明 |
|------|---------|---------|------|
| **系统内存 (RAM)** | 15GB (cgroup) | 🔴 **主要瓶颈** | 限制模型大小和数据量 |
| GPU 显存 (VRAM) | 12GB (RTX 3080 Ti) | 🟡 次要瓶颈 | 可通过 LoRA/梯度检查点缓解 |
| 单次作业时长 | 36 小时 | 🟢 可控 | 可通过 checkpoint 续训 |
| Kaggle 每日提交 | 10 次 | 🟢 可控 | 合理规划提交策略 |

### 瓶颈确认：是内存，不是显存

**关键证据**：
```
DeBERTa-v3-base (86M 参数) 训练时：
├── 系统内存 (RSS): ~12GB ← 接近 15GB 限制
├── GPU 显存: ~4-5GB ← 远未到 12GB 限制
└── 结论: 瓶颈是系统内存，不是 GPU 显存
```

**为什么内存是瓶颈**：
1. 模型加载到 CPU 内存（PyTorch 默认）
2. 数据预处理在 CPU 进行
3. 3M 条数据 × 文本特征 = 大量内存占用
4. HPC cgroup 限制是硬限制，超出会被 kill

### 各模型内存需求

| 模型 | 参数量 | 系统内存需求 | GPU 显存需求 | 当前可行? |
|------|--------|-------------|-------------|----------|
| DeBERTa-v3-base | 86M | ~12GB | ~4-5GB | ✅ 可行 (子样本) |
| DeBERTa-v3-base 全量 | 86M | ~20GB | ~4-5GB | ❌ 需提额 |
| DeBERTa-v3-large | 304M | ~22GB | ~6-8GB | ❌ 需提额 |
| RoBERTa-large | 355M | ~22GB | ~6-8GB | ❌ 需提额 |
| ELECTRA-large | 335M | ~22GB | ~6-8GB | ❌ 需提额 |

---

## 执行策略（带资源判断）

```
Phase 1: 立即执行 (无需额外资源)
├── Task 1: DeBERTa-v3-base 5-Fold 集成 [500K 数据]
├── Task 2: OOF 校准替代方差扩展
└── Task 3: 多折集成 + OOF 校准提交

Phase 2: 资源判断 (检查内存)
├── IF 内存 >= 20GB:
│   ├── Task 5: DeBERTa-v3-base 全量 3M × 5-Fold
│   └── Task 6: DeBERTa-v3-large (304M) 全量训练
├── ELSE (内存 < 20GB):
│   ├── Task 5a: DeBERTa-v3-base 子样本 × 10-Fold
│   ├── Task 5b: DeBERTa-v3-base 多种子集成
│   └── 提示用户申请内存提额

Phase 3: 高级优化 (资源到位后)
├── Task 7: 伪标签迭代
├── Task 8: 多架构集成 (RoBERTa, ELECTRA)
└── Task 9: 后处理优化 (等渗回归, 分位数校准)
```

---

## Phase 1: 立即执行 (1-2天，无需额外资源)

### Task 1: DeBERTa-v3-base 5-Fold 集成

**目标**: 用当前 500K 数据完成 5 折训练，集成预测

**步骤**:
1. 检查现有 checkpoint，确认哪些 fold 已完成
2. 补训剩余 fold (fold 2-5)
3. 集成 5 个 fold 的预测

**预期收益**: -0.01~0.02 RMSE

### Task 2: OOF 校准方案 (替代方差扩展)

**目标**: 用 OOF 预测的分布来校准，比方差扩展更稳健

**核心代码**:
```python
def oof_calibrate(oof_predictions, test_predictions, train_labels):
    """
    OOF 校准：用 OOF 预测的 mean 来中心化，用训练集标签的 std 来缩放
    
    Args:
        oof_predictions: K-Fold OOF 预测值
        test_predictions: 测试集预测值
        train_labels: 训练集标签
    
    Returns:
        校准后的测试集预测值
    """
    # 用 OOF 预测的 mean (承认模型的系统性偏差)
    oof_mean = oof_predictions.mean()
    oof_std = oof_predictions.std()
    
    # 用训练集标签的 mean/std 作为目标分布
    target_mean = train_labels.mean()
    target_std = train_labels.std()
    
    # 计算缩放因子
    scale = target_std / oof_std
    
    # 校准：中心化用 OOF mean，缩放用 target std
    pred_calibrated = (test_predictions - oof_mean) * scale + target_mean
    
    return pred_calibrated
```

**与方差扩展的区别**:
| 方法 | 中心化 | 缩放 | 稳健性 |
|------|--------|------|--------|
| 方差扩展 | pred_mean | target_std/pred_std | 中 |
| OOF 校准 | oof_mean | target_std/oof_std | 高 |

**为什么更稳健**:
- OOF 预测反映模型在未见数据上的真实行为
- 用 OOF mean 中心化，承认模型的系统性偏差
- 避免强行把预测拉到训练集 mean

**预期收益**: -0.005~0.01 RMSE (相比方差扩展)

### Task 3: 多折集成 + OOF 校准提交

**目标**: 组合 Task 1 和 Task 2，生成最佳提交

**方案**:
1. 5-Fold 预测取平均
2. 用 OOF 校准替代方差扩展
3. 尝试不同 blend 比例 (90/10, 85/15, 80/20)
4. 提交 Kaggle

**预期收益**: -0.015~0.03 RMSE (综合 Task 1+2)

---

## Phase 2: 资源判断与分支执行

### 资源检查脚本

```python
import psutil
import torch

def check_resources():
    """检查当前可用资源"""
    # 系统内存
    mem = psutil.virtual_memory()
    total_mem_gb = mem.total / (1024**3)
    available_mem_gb = mem.available / (1024**3)
    
    # GPU 显存
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        gpu_used_gb = torch.cuda.memory_allocated(0) / (1024**3)
        gpu_available_gb = gpu_mem_gb - gpu_used_gb
    else:
        gpu_mem_gb = 0
        gpu_available_gb = 0
    
    return {
        'total_mem_gb': total_mem_gb,
        'available_mem_gb': available_mem_gb,
        'gpu_mem_gb': gpu_mem_gb,
        'gpu_available_gb': gpu_available_gb,
        'can_train_large_model': total_mem_gb >= 20,
        'can_train_full_data': total_mem_gb >= 20,
    }

def print_resource_status():
    """打印资源状态和建议"""
    res = check_resources()
    
    print("=" * 60)
    print("资源状态检查")
    print("=" * 60)
    print(f"系统内存: {res['total_mem_gb']:.1f} GB (可用: {res['available_mem_gb']:.1f} GB)")
    print(f"GPU 显存: {res['gpu_mem_gb']:.1f} GB (可用: {res['gpu_available_gb']:.1f} GB)")
    print()
    
    if res['can_train_large_model']:
        print("✅ 资源充足，可以执行全量训练计划")
        print("   → 执行 Task 5: 全量 3M × 5-Fold")
        print("   → 执行 Task 6: DeBERTa-v3-large 训练")
    else:
        print("⚠️  内存不足，执行替代方案")
        print(f"   → 当前内存: {res['total_mem_gb']:.1f} GB")
        print(f"   → 需要内存: >= 20 GB")
        print()
        print("替代方案:")
        print("   → Task 5a: 子样本 × 10-Fold (降低方差)")
        print("   → Task 5b: 多种子集成 (增加多样性)")
        print()
        print("💡 建议: 申请 HPC 内存提额到 20-25GB")
        print("   申请理由: 训练 DeBERTa-v3-large 需要 ~22GB 内存")
    
    print("=" * 60)
    return res
```

### Task 4: 申请 HPC 内存提额

**当前限制**: 15GB (cgroup)
**需要**: 20-25GB

**申请理由**:
```
1. 项目进度: Kaggle RMSE 0.617 (竞赛前列)
2. 瓶颈: 系统内存 15GB 限制无法训练大模型
3. 需求: 20-25GB 训练 DeBERTa-v3-large (304M 参数)
4. 预期收益: RMSE 提升 3-5% (0.617 → 0.58-0.59)
5. 技术方案: LoRA + 梯度检查点 + 全量 3M 数据
```

**申请材料**:
- 当前项目进度 (Kaggle 0.617)
- 资源需求说明
- 预期收益
- 技术方案

---

### Task 5: 全量训练 (条件执行)

#### IF 内存 >= 20GB: Task 5 全量训练

**目标**: 用全量数据重新训练，最大化信息利用

**配置**:
- 数据: 全量 3M (每折 2.4M 训练 + 0.6M 验证)
- 模型: DeBERTa-v3-base + LoRA
- 折数: 5
- Epochs: 5 (early stopping patience=3)
- 预计时间: 5 折 × 3h = 15h

**预期收益**: -0.02~0.03 RMSE

#### ELSE (内存 < 20GB): Task 5a/5b 替代方案

**Task 5a: 子样本 × 10-Fold**

**目标**: 用更多折数降低方差，弥补数据量不足

**配置**:
- 数据: 500K 子样本
- 模型: DeBERTa-v3-base + LoRA
- 折数: 10 (而非 5)
- Epochs: 5
- 预计时间: 10 折 × 1.5h = 15h

**原理**:
- 更多折数 → 每个模型训练数据更多 (90% vs 80%)
- 更多模型 → 集成方差更低
- 预期效果: 接近全量 5 折的性能

**预期收益**: -0.015~0.025 RMSE

**Task 5b: 多种子集成**

**目标**: 用不同随机种子训练多个模型，增加多样性

**配置**:
- 数据: 500K 子样本
- 模型: DeBERTa-v3-base + LoRA
- 种子数: 3-5 个不同种子
- 每个种子: 5-Fold
- 预计时间: 3 种子 × 5 折 × 1.5h = 22.5h

**原理**:
- 不同种子 → 不同的 fold 划分
- 不同的训练轨迹 → 更好的多样性
- 集成效果: 降低方差，提高泛化

**预期收益**: -0.01~0.02 RMSE

---

### Task 6: DeBERTa-v3-large 训练 (条件执行)

#### IF 内存 >= 22GB:

**目标**: 用更大模型提升容量

**配置**:
- 模型: microsoft/deberta-v3-large (304M 参数)
- 数据: 全量 3M (或 1M 子样本)
- LoRA: r=16, alpha=32
- 预计内存: ~22GB
- 预计时间: 5 折 × 6h = 30h

**预期收益**: -0.03~0.05 RMSE

#### ELSE (内存 < 22GB):

**替代方案**: 继续用 DeBERTa-v3-base，但用更多折数和种子

**预期收益**: -0.015~0.025 RMSE

---

## Phase 3: 高级优化 (资源到位后)

### Task 7: 伪标签迭代

**目标**: 用最佳预测扩充训练数据

**步骤**:
1. 用当前最佳模型预测测试集
2. 选择高置信度预测 (如预测 std < 阈值)
3. 将测试集 + 伪标签加入训练集
4. 重新训练模型
5. 迭代 2-3 轮

**资源需求**: 取决于基础模型 (15-22GB)

**预期收益**: -0.02~0.03 RMSE

### Task 8: 多架构集成

**目标**: 用不同 Transformer 架构增加多样性

**候选模型**:
| 模型 | 参数量 | 内存需求 | 可行性 |
|------|--------|---------|--------|
| DeBERTa-v3-large | 304M | ~22GB | 需提额 |
| RoBERTa-large | 355M | ~22GB | 需提额 |
| ELECTRA-large | 335M | ~22GB | 需提额 |
| ALBERT-xxlarge | 235M | ~18GB | 勉强可行 |

**集成方式**:
- 各模型 5-Fold 预测
- Ridge stacking 或加权平均

**资源需求**: 22GB+ (每个大模型)

**预期收益**: -0.03~0.05 RMSE

### Task 9: 后处理优化

**目标**: 比方差扩展更好的校准方法

**候选方法**:
1. **等渗回归 (Isotonic Regression)**
   - 非参数校准，比线性缩放更灵活
   - 用 OOF 预测和标签拟合

2. **分位数校准 (Quantile Calibration)**
   - 将预测的分位数映射到目标分位数
   - 比均值/方差校准更精确

3. **温度缩放 (Temperature Scaling)**
   - 用于 CORAL 输出的概率校准
   - 单参数优化

**资源需求**: 低 (后处理，不涉及模型训练)

**预期收益**: -0.005~0.01 RMSE

---

## 成功标准

- [x] Kaggle < 0.62 (当前 0.61734)
- [ ] Kaggle < 0.60
- [ ] Kaggle < 0.58
- [ ] Kaggle < 0.55
- [ ] Kaggle < 0.50

---

## 时间线

| 阶段 | 时间 | 预期 RMSE | 累计提升 | 资源要求 |
|------|------|-----------|---------|---------|
| 当前 | - | 0.61734 | - | - |
| Phase 1 完成 | 1-2 天 | 0.60-0.61 | -2~3% | 15GB ✅ |
| Phase 2 完成 (替代) | 3-5 天 | 0.59-0.60 | -3~4% | 15GB ✅ |
| Phase 2 完成 (全量) | 1-2 周 | 0.57-0.59 | -5~7% | 20-25GB |
| Phase 3 完成 | 2-3 周 | 0.55-0.58 | -8~10% | 22GB+ |

---

## 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| HPC 内存申请被拒 | 中 | 高 | 用替代方案 (10-Fold + 多种子) |
| DeBERTa-large 训练失败 | 中 | 中 | 回退到 base + 更多折数 |
| 伪标签引入噪声 | 中 | 中 | 严格筛选置信度 |
| 过拟合测试集 | 低 | 高 | 控制提交频率，用 OOF 监控 |
| 内存超出被 kill | 中 | 中 | 监控内存使用，及时保存 checkpoint |

---

## 附录: 资源检查命令

```bash
# 检查系统内存
free -h

# 检查 cgroup 限制
cat /sys/fs/cgroup/memory/memory.limit_in_bytes

# 检查 GPU 显存
nvidia-smi

# Python 检查
python -c "import psutil; print(f'内存: {psutil.virtual_memory().total/(1024**3):.1f}GB')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_properties(0).total_mem/(1024**3):.1f}GB')"
```
