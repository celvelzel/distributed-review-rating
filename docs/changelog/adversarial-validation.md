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

**Overall OOF AUC: 0.5336**

> ✅ **No significant distribution shift** (AUC ≈ 0.5)

## Top-10 Features by Importance

| Rank | Feature | Importance (gain) |
|------|---------|-------------------|
| 1 | comment_len | 485.5 |
| 2 | review_year | 302.2 |
| 3 | title_len | 261.0 |
| 4 | review_hour | 187.5 |
| 5 | rating_number | 183.7 |
| 6 | price | 156.3 |
| 7 | review_month | 132.1 |
| 8 | votes | 103.6 |
| 9 | review_weekday | 95.3 |
| 10 | purchased | 26.0 |

## Recommendation

✅ No significant distribution shift (AUC=0.5336 ≈ 0.5). Train and test distributions are similar. Proceed with confidence.

## Notes
- This analysis is exploratory and does NOT modify the original data.
- Features used: votes, purchased, title_len, comment_len, price, rating_number, time features.
- `rating` is excluded from the classifier since it exists only in train.
- AUC ≈ 0.5 means train/test are indistinguishable (desired outcome).
