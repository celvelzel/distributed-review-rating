
## 2026-06-10: DeBERTa Fine-tuning Diagnosis (Task 1)

### Key Findings
- **val_rmse = 1.113** is worse than frozen embedding MLP (1.131), indicating fine-tuning is not learning effectively
- Root causes: suboptimal [CLS] pooling, insufficient epochs, no label normalization, wrong scheduler

### Issues Identified (6 total)
1. **[CLS] pooling** (HIGH): AutoModelForSequenceClassification uses [CLS] token, but DeBERTa doesn't use NSP → mean pooling is better for regression
2. **Only 3 epochs + patience=2** (HIGH): Can stop after 2 epochs; MLP trains 10 epochs with patience=10
3. **No label normalization** (MEDIUM-HIGH): Labels in [1,5] range cause large loss gradients; normalize to zero-mean/unit-variance
4. **Linear scheduler** (MEDIUM): Code uses linear decay but docstring says cosine; cosine decay is gentler and standard
5. **MAX_LENGTH=128** (MEDIUM): Truncates title+comment concatenation; increase to 256
6. **No discriminative LR** (MEDIUM): Same LR for all layers; should use 1e-5 backbone, 5e-5 head

### Expected Combined Improvement
- 15-25% RMSE reduction (from 1.113 to ~0.85-0.95)
- Quick wins first: epochs, label normalization, cosine scheduler
- Architecture changes: mean pooling, longer sequences

### Implementation Priority
1. Phase 1 (Low effort, High impact): epochs=10, patience=3, label normalization, cosine scheduler
2. Phase 2 (Medium effort, High impact): mean pooling custom model, MAX_LENGTH=256
3. Phase 3 (Medium effort, Medium impact): discriminative learning rates

### Evidence File
- Report: `.sisyphus/evidence/task-1-diagnosis-report.md`
