# Adversarial Validation Report — T11

**Date**: 2026-06-06

## Objective
Detect distribution shift between train and test sets using a LightGBM binary classifier.
If AUC ≈ 0.5 → distributions are similar (good). If AUC > 0.6 → shift detected.

## Method
- Labeled train=0, test=1
- Sampled 10K rows from each split for speed
- Trained LightGBM binary classifier with 5-fold stratified CV
- Measured OOF AUC and feature importance (gain)

## Results

**Overall OOF AUC: 0.5235**

> ✅ **No significant distribution shift** (AUC ≈ 0.5)

## Top-10 Features by Importance

| Rank | Feature | Importance (gain) |
|------|---------|-------------------|
| 1 | comment_len | 1038.0 |
| 2 | title_len | 720.4 |
| 3 | votes | 273.4 |
| 4 | purchased | 41.4 |
| 5 | price | 0.0 |
| 6 | rating_number | 0.0 |

## Recommendation

✅ No significant distribution shift (AUC=0.5235 ≈ 0.5). Train and test distributions are similar. Proceed with confidence.

## Notes
- This analysis is exploratory and does NOT modify the original data.
- Features used: votes, purchased, title_len, comment_len, price, rating_number, time features.
- `rating` is excluded from the classifier since it exists only in train.
- AUC ≈ 0.5 means train/test are indistinguishable (desired outcome).
