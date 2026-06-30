"""模型模块 — 包含 COMP5434 课程项目所有生产级预测模型。

模块分类:
  - DeBERTa 系列: deberta_lora_1m (1M子样本), deberta_base_full (3M全量), deberta_large_full (large)
  - 预测脚本: predict_lora_fold1, predict, submit_stacking_v3
  - 集成: stacking_v3 (9基模型 + 5元学习器自动选择)
  - GBDT 基模型: xgboost_train, tfidf_baseline, train_safe_features, train_graph_models
  - 神经网络: mlp (模型定义), run_mlp (训练入口)
  - 多样性集成: ensemble_diverse (LGB+XGB+MLP 加权平均)

所有模型输出 OOF 预测和测试预测 (.npy) 供 stacking_v3 集成使用。
"""
