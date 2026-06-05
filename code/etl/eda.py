#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis (EDA) for COMP5434 Review Rating Prediction.

Generates:
  - Console statistics (row counts, missing rates, distributions)
  - Visualization PNGs in docs/changelog/figures/
  - eda-report.md in docs/changelog/
  - metrics.json template in docs/changelog/

Usage:
    python code/etl/eda.py
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "docs" / "changelog" / "figures"
REPORT_PATH = ROOT / "docs" / "changelog" / "eda-report.md"
METRICS_PATH = ROOT / "docs" / "changelog" / "metrics.json"
NOTEPAD_PATH = ROOT / ".sisyphus" / "notepads" / "review-rating-iteration" / "learnings.md"

FIG_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "docs" / "changelog").mkdir(parents=True, exist_ok=True)
NOTEPAD_PATH.parent.mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.1)

# ─── Helper: save figure ─────────────────────────────────────────────────────
def savefig(name: str):
    path = FIG_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path.relative_to(ROOT)}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("COMP5434 EDA — Exploratory Data Analysis")
print("=" * 70)

# Use PySpark for full-dataset statistics
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, DoubleType

spark = (
    SparkSession.builder
    .appName("COMP5434_EDA")
    .master("local[*]")
    .config("spark.driver.memory", "8g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("\n[1/7] Loading datasets via PySpark ...")
train_df = spark.read.csv(str(DATA_DIR / "train.csv"), header=True, inferSchema=True)
test_df  = spark.read.csv(str(DATA_DIR / "test.csv"),  header=True, inferSchema=True)
prod_df  = spark.read.csv(str(DATA_DIR / "prodInfo.csv"), header=True, inferSchema=True)

# Cast rating and votes to numeric (CSV has multi-line comments causing misalignment)
train_df = train_df.withColumn("rating", F.col("rating").cast("double"))
train_df = train_df.withColumn("votes", F.col("votes").cast("double"))
train_df = train_df.filter(F.col("rating").isNotNull() & F.col("rating").between(1, 5))
train_df = train_df.filter(F.col("votes").isNotNull())
test_df = test_df.withColumn("votes", F.col("votes").cast("double"))

# ─── 1a. Row counts ──────────────────────────────────────────────────────────
print("\n[2/7] Basic counts ...")
train_count = train_df.count()
test_count  = test_df.count()
prod_count  = prod_df.count()
print(f"  train.csv    : {train_count:>10,} rows")
print(f"  test.csv     : {test_count:>10,} rows")
print(f"  prodInfo.csv : {prod_count:>10,} rows")

# ─── 1b. Schemas ─────────────────────────────────────────────────────────────
print("\n  train schema:")
train_df.printSchema()
print("  test schema:")
test_df.printSchema()
print("  prodInfo schema:")
prod_df.printSchema()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. MISSING RATE PER COLUMN
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Missing rate per column ...")

def missing_rate_sdf(df, name):
    """Compute missing (null + empty string) rate for each column."""
    total = df.count()
    exprs = []
    for c in df.columns:
        exprs.append(
            F.round(
                F.sum(
                    F.when(
                        F.col(c).isNull() | (F.trim(F.col(c)) == ""), 1
                    ).otherwise(0)
                ) / total * 100, 2
            ).alias(c)
        )
    result = df.agg(*exprs).toPandas().T
    result.columns = ["missing_%"]
    result.index.name = "column"
    print(f"\n  --- {name} missing rate (%) ---")
    print(result.to_string())
    return result

train_missing = missing_rate_sdf(train_df, "train")
test_missing  = missing_rate_sdf(test_df, "test")
prod_missing  = missing_rate_sdf(prod_df, "prodInfo")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. RATING DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Rating distribution ...")

rating_dist = (
    train_df.groupBy("rating")
    .count()
    .orderBy("rating")
    .toPandas()
)
# Round rating to integer for clean grouping
rating_dist["rating"] = rating_dist["rating"].round().astype(int)
rating_dist = rating_dist.groupby("rating", as_index=False)["count"].sum()
rating_dist["pct"] = (rating_dist["count"] / rating_dist["count"].sum() * 100).round(2)
print(rating_dist.to_string(index=False))

# ── Plot 1: Rating distribution bar chart ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = sns.color_palette("viridis", len(rating_dist))
bars = ax.bar(rating_dist["rating"].astype(str), rating_dist["count"], color=colors, edgecolor="white")
for bar, pct in zip(bars, rating_dist["pct"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20000,
            f"{pct}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_xlabel("Rating", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Rating Distribution (Train Set)", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M"))
savefig("eda-rating-distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. REVIEW LENGTH DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Review length distribution ...")

# Use pandas for string-length calculations (sample 200K for speed)
SAMPLE_N = 200_000
train_sample = train_df.limit(SAMPLE_N).toPandas()

# Ensure numeric types in pandas
train_sample["rating"] = pd.to_numeric(train_sample["rating"], errors="coerce")
train_sample["votes"] = pd.to_numeric(train_sample["votes"], errors="coerce")
train_sample = train_sample.dropna(subset=["rating", "votes"])

train_sample["title_len"] = train_sample["title"].fillna("").str.len()
train_sample["comment_len"] = train_sample["comment"].fillna("").str.len()

for col_name in ["title_len", "comment_len"]:
    s = train_sample[col_name]
    print(f"\n  {col_name}:")
    print(f"    mean={s.mean():.1f}  median={s.median():.1f}  std={s.std():.1f}  "
          f"min={s.min()}  max={s.max()}")

# ── Plot 2: Review length histogram ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train_sample["title_len"], bins=80, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].set_title("Title Length Distribution", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Characters")
axes[0].set_ylabel("Count")
axes[0].axvline(train_sample["title_len"].median(), color="red", ls="--", label=f"median={train_sample['title_len'].median():.0f}")
axes[0].legend()

axes[1].hist(train_sample["comment_len"], bins=200, color="coral", edgecolor="white", alpha=0.85)
axes[1].set_title("Comment Length Distribution", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Characters")
axes[1].set_ylabel("Count")
axes[1].axvline(train_sample["comment_len"].median(), color="red", ls="--", label=f"median={train_sample['comment_len'].median():.0f}")
axes[1].legend()

plt.suptitle("Review Length Distributions (Sample 200K)", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
savefig("eda-review-length.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. UNIQUE COUNTS & OVERLAP
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] Unique counts & train/test overlap ...")

train_users = set(train_df.select("user_id").distinct().toPandas()["user_id"])
test_users  = set(test_df.select("user_id").distinct().toPandas()["user_id"])
train_prods = set(train_df.select("prod_id").distinct().toPandas()["prod_id"])
test_prods  = set(test_df.select("prod_id").distinct().toPandas()["prod_id"])
train_parent = set(train_df.select("parent_prod_id").distinct().toPandas()["parent_prod_id"])
prod_parent  = set(prod_df.select("parent_prod_id").distinct().toPandas()["parent_prod_id"])

print(f"  Unique train user_id       : {len(train_users):>10,}")
print(f"  Unique test  user_id       : {len(test_users):>10,}")
print(f"  Unique train prod_id       : {len(train_prods):>10,}")
print(f"  Unique test  prod_id       : {len(test_prods):>10,}")
print(f"  Unique train parent_prod_id: {len(train_parent):>10,}")
print(f"  Unique prodInfo parent_prod_id: {len(prod_parent):>10,}")

user_overlap = train_users & test_users
prod_overlap = train_prods & test_prods
cold_users   = test_users - train_users

user_overlap_rate = len(user_overlap) / len(test_users) * 100
prod_overlap_rate = len(prod_overlap) / len(test_prods) * 100
cold_user_ratio   = len(cold_users) / len(test_users) * 100

print(f"\n  Train/Test user overlap    : {len(user_overlap):>6,} / {len(test_users):>6,} = {user_overlap_rate:.2f}%")
print(f"  Train/Test prod overlap    : {len(prod_overlap):>6,} / {len(test_prods):>6,} = {prod_overlap_rate:.2f}%")
print(f"  Cold-start users (test\\train): {len(cold_users):>6,} / {len(test_users):>6,} = {cold_user_ratio:.2f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. TIME RANGE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n  Time range (train) ...")
time_stats = train_df.agg(
    F.min("time").alias("earliest_ms"),
    F.max("time").alias("latest_ms"),
    F.min(F.from_unixtime(F.col("time") / 1000)).alias("earliest_iso"),
    F.max(F.from_unixtime(F.col("time") / 1000)).alias("latest_iso"),
).toPandas().iloc[0]

print(f"    Earliest : {time_stats['earliest_iso']}  (ms={int(time_stats['earliest_ms'])})")
print(f"    Latest   : {time_stats['latest_iso']}  (ms={int(time_stats['latest_ms'])})")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. VOTES & PURCHASED DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n  Votes distribution ...")
votes_stats = train_df.agg(
    F.mean("votes").alias("mean"),
    F.expr("percentile_approx(votes, 0.5)").alias("median"),
    F.stddev("votes").alias("std"),
    F.min("votes").alias("min"),
    F.max("votes").alias("max"),
    F.sum(F.when(F.col("votes") == 0, 1).otherwise(0)).alias("zeros"),
).toPandas().iloc[0]
print(f"    mean={votes_stats['mean']:.2f}  median={votes_stats['median']}  "
      f"std={votes_stats['std']:.2f}  min={votes_stats['min']}  max={votes_stats['max']}  "
      f"zeros={int(votes_stats['zeros']):,}")

print("\n  Purchased distribution ...")
purchased_dist = train_df.groupBy("purchased").count().toPandas()
purchased_dist["pct"] = (purchased_dist["count"] / purchased_dist["count"].sum() * 100).round(2)
print(purchased_dist.to_string(index=False))

# ── Plot 5: Votes vs Rating scatter (sampled) ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
scatter_data = train_sample[["rating", "votes"]].copy()
scatter_data = scatter_data.dropna()
# Add jitter for visualization
rng = np.random.default_rng(42)
scatter_data["rating_j"] = scatter_data["rating"] + rng.uniform(-0.15, 0.15, len(scatter_data))
ax.scatter(scatter_data["rating_j"], scatter_data["votes"], alpha=0.05, s=8, color="steelblue")
ax.set_xlabel("Rating", fontsize=12)
ax.set_ylabel("Votes", fontsize=12)
ax.set_title("Votes vs Rating (Sample 200K)", fontsize=14, fontweight="bold")
ax.set_xticks([1, 2, 3, 4, 5])
savefig("eda-votes-vs-rating.png")

# ── Plot 3: Rating over time trend ───────────────────────────────────────────
print("\n  Rating over time trend ...")
train_sample["time"] = pd.to_numeric(train_sample["time"], errors="coerce")
train_sample = train_sample.dropna(subset=["time"])
train_sample["year"] = pd.to_datetime(train_sample["time"], unit="ms").dt.year
yearly_rating = train_sample.groupby("year")["rating"].agg(["mean", "count"]).reset_index()
yearly_rating = yearly_rating[yearly_rating["count"] >= 100]  # filter sparse years

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(yearly_rating["year"], yearly_rating["mean"], "o-", color="steelblue", lw=2, label="Mean rating")
ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Mean Rating", fontsize=12, color="steelblue")
ax1.set_ylim(3.5, 4.8)
ax2 = ax1.twinx()
ax2.bar(yearly_rating["year"], yearly_rating["count"], alpha=0.3, color="gray", label="Review count")
ax2.set_ylabel("Review Count", fontsize=12, color="gray")
ax1.set_title("Rating Trend Over Time (Sample 200K)", fontsize=14, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
savefig("eda-rating-over-time.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. PRODINFO ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n  ProdInfo analysis ...")

# Top 10 categories
cat_dist = (
    prod_df.groupBy("main_category")
    .count()
    .orderBy(F.desc("count"))
    .limit(15)
    .toPandas()
)
print("  Top 15 categories:")
print(cat_dist.to_string(index=False))

# ── Plot 4: Top 10 categories ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
top10 = cat_dist.head(10).copy()
top10["main_category"] = top10["main_category"].fillna("(unknown)").astype(str)
top10 = top10.iloc[::-1]  # reverse for horizontal bar
colors = sns.color_palette("viridis", len(top10))
ax.barh(top10["main_category"], top10["count"], color=colors, edgecolor="white")
ax.set_xlabel("Number of Products", fontsize=12)
ax.set_title("Top 10 Product Categories (prodInfo)", fontsize=14, fontweight="bold")
for i, (v, cat) in enumerate(zip(top10["count"], top10["main_category"])):
    ax.text(v + 200, i, f"{v:,}", va="center", fontsize=10)
savefig("eda-top-categories.png")

# ── Plot 6: Price distribution ───────────────────────────────────────────────
print("\n  Price distribution ...")
prod_sample = prod_df.filter(F.col("price").isNotNull()).limit(100_000).toPandas()
# price column might be string, convert
prod_sample["price_num"] = pd.to_numeric(prod_sample["price"], errors="coerce")
price_valid = prod_sample["price_num"].dropna()

print(f"    Products with price: {len(price_valid):,} / {len(prod_sample):,}")
if len(price_valid) > 0:
    print(f"    mean=${price_valid.mean():.2f}  median=${price_valid.median():.2f}  "
          f"std=${price_valid.std():.2f}  min=${price_valid.min():.2f}  max=${price_valid.max():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full distribution (clipped at 99th percentile for readability)
    p99 = price_valid.quantile(0.99)
    axes[0].hist(price_valid.clip(upper=p99), bins=80, color="seagreen", edgecolor="white", alpha=0.85)
    axes[0].set_title("Price Distribution (clipped at 99th pct)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Price ($)")
    axes[0].set_ylabel("Count")
    axes[0].axvline(price_valid.median(), color="red", ls="--", label=f"median=${price_valid.median():.2f}")
    axes[0].legend()

    # Log price
    log_price = np.log1p(price_valid)
    axes[1].hist(log_price, bins=80, color="salmon", edgecolor="white", alpha=0.85)
    axes[1].set_title("Log(1+Price) Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("log(1 + price)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    savefig("eda-price-distribution.png")
else:
    print("    [WARN] No valid prices found!")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. RATINGS BY CATEGORY (join train ↔ prodInfo)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n  Rating by category (sample join) ...")
train_cat_df = train_df.join(prod_df.select("parent_prod_id", "main_category"), on="parent_prod_id", how="left")
cat_rating = (
    train_cat_df.groupBy("main_category")
    .agg(F.avg(F.col("rating")).alias("mean_rating"), F.count("*").alias("n"))
    .filter(F.col("n") >= 1000)
    .orderBy(F.desc("n"))
    .toPandas()
)
if len(cat_rating) > 0:
    print(cat_rating.head(15).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 10. RATING_NUMBER (from prodInfo)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n  rating_number (prodInfo) ...")
prod_df = prod_df.withColumn("rating_number", F.col("rating_number").cast("double"))
rn_stats = prod_df.agg(
    F.mean("rating_number").alias("mean"),
    F.expr("percentile_approx(rating_number, 0.5)").alias("median"),
    F.min("rating_number").alias("min"),
    F.max("rating_number").alias("max"),
).toPandas().iloc[0]
print(f"    mean={rn_stats['mean']:.2f}  median={rn_stats['median']}  "
      f"min={rn_stats['min']}  max={rn_stats['max']}")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. GENERATE REPORT (eda-report.md)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Generating report & metrics.json ...")

report = f"""# EDA Report — COMP5434 Review Rating Prediction

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 1. Dataset Sizes

| Dataset    | Rows        |
|------------|-------------|
| train.csv  | {train_count:>12,} |
| test.csv   | {test_count:>12,} |
| prodInfo.csv | {prod_count:>12,} |

## 2. Schema

**train.csv**: id, user_id, prod_id, parent_prod_id, title, comment, time, votes, purchased, rating

**test.csv**: id, user_id, prod_id, parent_prod_id, title, comment, time, votes, purchased

**prodInfo.csv**: id, parent_prod_id, main_category, price, title, features, store, rating_number

## 3. Missing Rate (%)

### train.csv
{train_missing.to_markdown()}

### test.csv
{test_missing.to_markdown()}

### prodInfo.csv
{prod_missing.to_markdown()}

## 4. Rating Distribution

| Rating | Count      | Pct (%) |
|--------|------------|---------|
"""

for _, row in rating_dist.iterrows():
    report += f"| {int(row['rating'])} | {int(row['count']):>10,} | {row['pct']:.2f} |\n"

report += f"""
![Rating Distribution](figures/eda-rating-distribution.png)

## 5. Review Length Statistics

| Metric | Title Length | Comment Length |
|--------|-------------|----------------|
| mean   | {train_sample['title_len'].mean():.1f} | {train_sample['comment_len'].mean():.1f} |
| median | {train_sample['title_len'].median():.1f} | {train_sample['comment_len'].median():.1f} |
| std    | {train_sample['title_len'].std():.1f} | {train_sample['comment_len'].std():.1f} |
| min    | {train_sample['title_len'].min()} | {train_sample['comment_len'].min()} |
| max    | {train_sample['title_len'].max()} | {train_sample['comment_len'].max()} |

![Review Length Distribution](figures/eda-review-length.png)

## 6. Unique Counts & Overlap

| Metric | Value |
|--------|-------|
| Unique train user_id | {len(train_users):,} |
| Unique test user_id  | {len(test_users):,} |
| Unique train prod_id | {len(train_prods):,} |
| Unique test prod_id  | {len(test_prods):,} |
| Unique train parent_prod_id | {len(train_parent):,} |
| Unique prodInfo parent_prod_id | {len(prod_parent):,} |
| User overlap (train ∩ test) | {len(user_overlap):,} / {len(test_users):,} = {user_overlap_rate:.2f}% |
| Product overlap (train ∩ test) | {len(prod_overlap):,} / {len(test_prods):,} = {prod_overlap_rate:.2f}% |
| Cold-start users (test \\ train) | {len(cold_users):,} / {len(test_users):,} = {cold_user_ratio:.2f}% |

## 7. Time Range

| Metric | Value |
|--------|-------|
| Earliest | {time_stats['earliest_iso']} |
| Latest   | {time_stats['latest_iso']} |

## 8. Votes Distribution

| Metric | Value |
|--------|-------|
| mean   | {votes_stats['mean']:.2f} |
| median | {int(votes_stats['median'])} |
| std    | {votes_stats['std']:.2f} |
| min    | {int(votes_stats['min'])} |
| max    | {int(votes_stats['max'])} |
| zeros  | {int(votes_stats['zeros']):,} |

![Votes vs Rating](figures/eda-votes-vs-rating.png)

## 9. Purchased Distribution

| Purchased | Count      | Pct (%) |
|-----------|------------|---------|
"""

for _, row in purchased_dist.iterrows():
    report += f"| {row['purchased']} | {int(row['count']):>10,} | {row['pct']:.2f} |\n"

report += f"""
## 10. Rating Over Time

![Rating Over Time](figures/eda-rating-over-time.png)

## 11. Top Product Categories (prodInfo)

![Top 10 Categories](figures/eda-top-categories.png)

## 12. Price Distribution

"""

if len(price_valid) > 0:
    report += f"""| Metric | Value |
|--------|-------|
| Products with price | {len(price_valid):,} / {len(prod_sample):,} |
| mean   | ${price_valid.mean():.2f} |
| median | ${price_valid.median():.2f} |
| std    | ${price_valid.std():.2f} |
| min    | ${price_valid.min():.2f} |
| max    | ${price_valid.max():.2f} |

![Price Distribution](figures/eda-price-distribution.png)
"""

report += f"""
## 13. Key Findings

1. **Severe class imbalance**: Rating 5 dominates (~{rating_dist[rating_dist['rating']==5]['pct'].values[0]:.0f}%), ratings 1-2 are minority classes.
2. **Cold-start problem**: {cold_user_ratio:.1f}% of test users never appear in training data.
3. **User overlap**: Only {user_overlap_rate:.1f}% of test users are in training — limited user-level features.
4. **Product overlap**: {prod_overlap_rate:.1f}% of test products appear in training — product-level features are viable.
5. **Reviews are short**: Median title ~{train_sample['title_len'].median():.0f} chars, median comment ~{train_sample['comment_len'].median():.0f} chars.
6. **Most reviews have 0 votes**: Median votes={int(votes_stats['median'])}, highly skewed.
7. **Purchased dominates**: ~{purchased_dist[purchased_dist['purchased']==True]['pct'].values[0] if len(purchased_dist[purchased_dist['purchased']==True]) > 0 else 'N/A'}% of reviews are from verified purchases.
8. **Price data sparse**: Only {len(price_valid):,} / {len(prod_sample):,} products have price info.
9. **Data timespan**: Reviews span {time_stats['earliest_iso'][:4]} to {time_stats['latest_iso'][:4]}.
10. **Rating_number**: Products have on average {rn_stats['mean']:.0f} prior ratings (median={int(rn_stats['median'])}).
"""

REPORT_PATH.write_text(report, encoding="utf-8")
print(f"  [saved] {REPORT_PATH.relative_to(ROOT)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. METRICS.JSON TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════
metrics = {
    "project": "COMP5434 Review Rating Prediction",
    "team": "TBD",
    "hardware": "Local single GPU + PySpark pseudo-distributed",
    "stages": {
        "0": {"rmse": None, "train_time_sec": None, "inference_time_sec": None,
              "model": "tfidf_lgb", "features": ["tfidf"]},
        "1": {"rmse": None, "train_time_sec": None, "inference_time_sec": None,
              "model": "lgb_stats", "features": ["tfidf", "user_stats", "prod_stats", "temporal"]},
        "2": {"rmse": None, "train_time_sec": None, "inference_time_sec": None,
              "model": "lgb_multimodal", "features": ["tfidf", "user_stats", "prod_stats", "temporal", "bert", "lightgcn"]},
        "3": {"rmse": None, "train_time_sec": None, "inference_time_sec": None,
              "model": "lgb_multimodal", "features": ["tfidf", "user_stats", "prod_stats", "temporal", "bert", "lightgcn"]},
        "4": {"rmse": None, "train_time_sec": None, "inference_time_sec": None,
              "model": "stacking", "features": ["all"]},
        "5": {"rmse": None, "train_time_sec": None, "inference_time_sec": None,
              "model": "stacking_tuned", "features": ["all"]}
    },
    "ablations": {
        "a_no_text":        {"rmse": None, "delta_vs_full": None},
        "b_no_graph":       {"rmse": None, "delta_vs_full": None},
        "c_no_stacking":    {"rmse": None, "delta_vs_full": None},
        "d_no_kfold_te":    {"rmse": None, "delta_vs_full": None},
        "e_tfidf_vs_bert":  {"rmse": None, "delta_vs_full": None},
        "f_no_clip":        {"rmse": None, "delta_vs_full": None}
    }
}

METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"  [saved] {METRICS_PATH.relative_to(ROOT)}")

# ═══════════════════════════════════════════════════════════════════════════════
# 13. APPEND TO NOTEPAD
# ═══════════════════════════════════════════════════════════════════════════════
learnings = f"""

## EDA Learnings ({datetime.now().strftime("%Y-%m-%d")})

### Dataset Structure
- train: {train_count:,} rows, 10 columns; test: {test_count:,} rows, 9 columns; prodInfo: {prod_count:,} rows, 8 columns
- rating is integer 1-5, target variable
- time is Unix timestamp in milliseconds

### Class Imbalance
- Rating 5 is dominant class (~{rating_dist[rating_dist['rating']==5]['pct'].values[0]:.0f}%)
- Ratings 1-2 are minority — need stratified sampling or class weighting

### User/Product Overlap
- User overlap: {user_overlap_rate:.1f}% — very cold-start heavy
- Product overlap: {prod_overlap_rate:.1f}% — product features more generalizable
- {cold_user_ratio:.1f}% cold-start users in test

### Text Features
- Median title length: ~{train_sample['title_len'].median():.0f} chars
- Median comment length: ~{train_sample['comment_len'].median():.0f} chars
- TF-IDF and BERT embeddings will be key features

### Votes & Purchased
- Median votes: {int(votes_stats['median'])}, highly skewed (most reviews have 0 votes)
- Purchased is binary, majority True — weak feature alone but could interact

### Price Data
- Only {len(price_valid):,} / {len(prod_sample):,} products have price info — sparse
- Prices are right-skewed, consider log transform

### Implications for Modeling
1. Need robust handling of cold-start users (content-based fallback)
2. Product-level features (category, price, rating_number) are important
3. Text features (title + comment) are critical given short reviews
4. Temporal features could capture evolving review patterns
5. Class imbalance → consider focal loss or stratified K-fold
"""

if NOTEPAD_PATH.exists():
    existing = NOTEPAD_PATH.read_text(encoding="utf-8")
    NOTEPAD_PATH.write_text(existing + learnings, encoding="utf-8")
else:
    NOTEPAD_PATH.write_text("# Review Rating Iteration — Learnings\n" + learnings, encoding="utf-8")
print(f"  [saved] {NOTEPAD_PATH.relative_to(ROOT)}")

# ─── Cleanup ──────────────────────────────────────────────────────────────────
spark.stop()
print("\n" + "=" * 70)
print("EDA COMPLETE — all outputs generated successfully.")
print("=" * 70)
