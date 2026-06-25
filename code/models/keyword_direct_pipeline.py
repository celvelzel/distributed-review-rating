#!/usr/bin/env python
"""Leakage-safe keyword direct-pair pipeline for title -> rating overrides.

This script replaces weak heuristic keyword matching with an OOF-validated
keyword pair miner and a controlled merge step for advanced submission CSVs.

Workflow:
1) Learn direct keyword-rating pairs from train titles with K-fold validation.
2) Apply approved pairs to test titles to build direct keyword-rate predictions.
3) Optionally merge those direct predictions into an advanced model submission.

Example:
  ./.venv/bin/python code/models/keyword_direct_pipeline.py \
    --keywords-file data/test_candidate_keywords.txt \
    --base-submission output/sub-deb1m-ve60-sv3rlg40.csv \
    --rules-out data/direct_keyword_pairs.csv \
    --direct-out data/direct_keyword_rate.csv \
        --merged-out output/submissiona/sub-deb1m-ve60-sv3rlg40-keyword.csv
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class RuleThresholds:
    min_global_matches: int
    min_oof_matches: int
    min_oof_precision: float
    min_margin_pct: float
    min_lift: float


def _resolve_existing_path(user_path: str | None, candidates: list[str], label: str) -> Path:
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
        return path

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path

    raise FileNotFoundError(f"Cannot find {label}. Checked: {', '.join(candidates)}")


def _resolve_base_submission_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path

    # Fallback: look up same filename in common archive locations.
    candidates = [
        Path("output") / path.name,
        Path("output/archive/submissions") / path.name,
        Path("output/archive") / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            print(f"Base submission not found at requested path, using fallback: {candidate}")
            return candidate

    # Final fallback: fuzzy match by stem in archive submissions.
    archive_dir = Path("output/archive/submissions")
    if archive_dir.exists():
        nearby = sorted(archive_dir.glob(f"*{path.stem}*.csv"))
        if nearby:
            preview = ", ".join(str(p) for p in nearby[:5])
            raise FileNotFoundError(
                "base submission not found: "
                f"{path}. Similar files in output/archive/submissions: {preview}"
            )

    raise FileNotFoundError(
        "base submission not found: "
        f"{path}. Also checked output/, output/archive/submissions/, output/archive/."
    )


def _normalize_title(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _extract_direct_star_signal(title_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    patterns = {
        5: [r"\b5\s*[- ]?stars?\b", r"\bfive\s*[- ]?stars?\b"],
        4: [r"\b4\s*[- ]?stars?\b", r"\bfour\s*[- ]?stars?\b"],
        3: [r"\b3\s*[- ]?stars?\b", r"\bthree\s*[- ]?stars?\b"],
        2: [r"\b2\s*[- ]?stars?\b", r"\btwo\s*[- ]?stars?\b"],
        1: [r"\b1\s*[- ]?stars?\b", r"\bone\s*[- ]?stars?\b"],
    }

    s = _normalize_title(title_series)
    claimed = pd.Series(np.nan, index=s.index, dtype="float64")
    multi_claim = pd.Series(False, index=s.index)

    for stars, regex_list in patterns.items():
        hit = pd.Series(False, index=s.index)
        for regex in regex_list:
            hit = hit | s.str.contains(regex, regex=True, na=False)
        multi_claim = multi_claim | (hit & claimed.notna())
        claimed = claimed.mask(hit & claimed.isna(), float(stars))

    return claimed, multi_claim


def _build_keyword_regex(keyword: str) -> str:
    parts = [p for p in re.split(r"\s+", keyword.strip().lower()) if p]
    escaped = [re.escape(part) for part in parts]
    joined = r"\s+".join(escaped)
    return rf"\b{joined}\b"


def _load_keywords(args: argparse.Namespace) -> list[str]:
    keywords: list[str] = []

    if args.keywords:
        keywords.extend(args.keywords)

    if args.keywords_file:
        file_path = Path(args.keywords_file)
        if not file_path.exists():
            raise FileNotFoundError(f"keywords file not found: {file_path}")
        for line in file_path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                keywords.append(value)

    cleaned = [k.strip().lower() for k in keywords if k and k.strip()]
    cleaned = list(dict.fromkeys(cleaned))

    direct_terms = {
        "1", "2", "3", "4", "5", "one", "two", "three", "four", "five", "star", "stars"
    }
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "in", "is", "it",
        "its", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "will", "with",
        "what", "when", "where", "who", "why", "how", "all", "any", "can", "dont", "i", "im", "ive",
        "more", "most", "much", "new", "now", "only", "other", "out", "over", "some", "still", "than",
        "there", "these", "they", "too", "very", "way",
    }

    filtered: list[str] = []
    for keyword in cleaned:
        if keyword in direct_terms:
            continue
        if keyword in stopwords:
            continue
        if re.fullmatch(r"(?:[1-5]|one|two|three|four|five)\s*[- ]?stars?", keyword):
            continue
        if len(keyword) < args.min_keyword_len:
            continue
        filtered.append(keyword)

    if args.max_keywords is not None:
        filtered = filtered[: args.max_keywords]

    if not filtered:
        raise ValueError("No usable keywords were provided after filtering.")

    return filtered


def _safe_rating(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().clip(1, 5).astype("Int64")


def _top_two_counts(rating_series: pd.Series) -> tuple[int, float, float, float]:
    counts = rating_series.value_counts()
    total = int(counts.sum())
    if total == 0:
        return 0, 0.0, 0.0, 0.0

    top = counts.iloc[0]
    second = counts.iloc[1] if len(counts) > 1 else 0
    dominant_pct = float(top) / total * 100.0
    second_pct = float(second) / total * 100.0
    margin = dominant_pct - second_pct
    return int(counts.index[0]), dominant_pct, second_pct, margin


def _iter_keyword_masks(title_series: pd.Series, keywords: Iterable[str]) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    for keyword in keywords:
        regex = _build_keyword_regex(keyword)
        masks[keyword] = title_series.str.contains(regex, regex=True, na=False)
    return masks


def mine_rules(train_df: pd.DataFrame, test_df: pd.DataFrame, keywords: list[str], args: argparse.Namespace) -> pd.DataFrame:
    required_train = {"title", "rating"}
    required_test = {"id", "title"}

    missing_train = required_train - set(train_df.columns)
    missing_test = required_test - set(test_df.columns)
    if missing_train:
        raise ValueError(f"train.csv missing columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"test.csv missing columns: {sorted(missing_test)}")

    train = train_df.copy()
    test = test_df.copy()
    train["title_norm"] = _normalize_title(train["title"])
    test["title_norm"] = _normalize_title(test["title"])
    train["rating_bucket"] = _safe_rating(train["rating"])
    train = train[train["rating_bucket"].notna()].copy()

    y = train["rating_bucket"].astype(int)
    baseline = y.value_counts(normalize=True).to_dict()
    baseline_pct = {r: float(baseline.get(r, 0.0) * 100.0) for r in range(1, 6)}

    train_masks = _iter_keyword_masks(train["title_norm"], keywords)
    test_masks = _iter_keyword_masks(test["title_norm"], keywords)

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    rows: list[dict[str, object]] = []
    for idx, keyword in enumerate(keywords, start=1):
        print(f"[{idx}/{len(keywords)}] mining: {keyword}")
        full_mask = train_masks[keyword]
        test_mask = test_masks[keyword]
        matched_all = train.loc[full_mask, "rating_bucket"].astype(int)

        global_match_count = int(full_mask.sum())
        test_match_count = int(test_mask.sum())

        if global_match_count == 0:
            rows.append(
                {
                    "keyword": keyword,
                    "approved": 0,
                    "reject_reason": "no_train_match",
                    "global_train_matches": 0,
                    "test_matches": test_match_count,
                    "dominant_rating": None,
                    "global_dominant_pct": 0.0,
                    "global_margin_pct": 0.0,
                    "global_lift": 0.0,
                    "oof_matches": 0,
                    "oof_precision": 0.0,
                    "oof_rating_consistency": 0.0,
                    "rule_score": 0.0,
                }
            )
            continue

        dominant_rating, dom_pct, _second_pct, margin_pct = _top_two_counts(matched_all)
        dominant_baseline = baseline_pct.get(dominant_rating, 0.0)
        lift = dom_pct / dominant_baseline if dominant_baseline > 0 else 0.0

        fold_oof_matches = 0
        fold_oof_correct = 0
        fold_pred_ratings: list[int] = []

        for tr_idx, va_idx in skf.split(train, y):
            tr = train.iloc[tr_idx]
            va = train.iloc[va_idx]
            tr_mask = train_masks[keyword].iloc[tr_idx]
            va_mask = train_masks[keyword].iloc[va_idx]

            tr_hits = tr.loc[tr_mask, "rating_bucket"].astype(int)
            if tr_hits.empty:
                continue

            fold_rating, _fold_dom_pct, _fold_second_pct, _fold_margin = _top_two_counts(tr_hits)
            fold_pred_ratings.append(fold_rating)

            va_hits = va.loc[va_mask, "rating_bucket"].astype(int)
            if va_hits.empty:
                continue

            fold_oof_matches += len(va_hits)
            fold_oof_correct += int((va_hits == fold_rating).sum())

        oof_precision = (fold_oof_correct / fold_oof_matches) if fold_oof_matches > 0 else 0.0
        consistency = 0.0
        if fold_pred_ratings:
            pred_counts = pd.Series(fold_pred_ratings).value_counts(normalize=True)
            consistency = float(pred_counts.max())

        thresholds = RuleThresholds(
            min_global_matches=args.min_global_matches,
            min_oof_matches=args.min_oof_matches,
            min_oof_precision=args.min_oof_precision,
            min_margin_pct=args.min_margin_pct,
            min_lift=args.min_lift,
        )

        approved = 1
        reject_reason = "approved"
        if global_match_count < thresholds.min_global_matches:
            approved = 0
            reject_reason = "low_global_support"
        elif fold_oof_matches < thresholds.min_oof_matches:
            approved = 0
            reject_reason = "low_oof_support"
        elif oof_precision < thresholds.min_oof_precision:
            approved = 0
            reject_reason = "low_oof_precision"
        elif margin_pct < thresholds.min_margin_pct:
            approved = 0
            reject_reason = "low_margin"
        elif lift < thresholds.min_lift:
            approved = 0
            reject_reason = "low_lift"
        elif consistency < args.min_oof_consistency:
            approved = 0
            reject_reason = "unstable_fold_label"

        score = (
            oof_precision
            * (margin_pct / 100.0)
            * math.log1p(global_match_count)
            * max(lift, 1e-9)
        )

        rows.append(
            {
                "keyword": keyword,
                "approved": approved,
                "reject_reason": reject_reason,
                "dominant_rating": int(dominant_rating),
                "global_train_matches": global_match_count,
                "test_matches": test_match_count,
                "global_dominant_pct": round(dom_pct, 6),
                "global_margin_pct": round(margin_pct, 6),
                "global_lift": round(lift, 6),
                "oof_matches": int(fold_oof_matches),
                "oof_precision": round(oof_precision, 6),
                "oof_rating_consistency": round(consistency, 6),
                "rule_score": round(score, 8),
                "baseline_pct_dominant": round(dominant_baseline, 6),
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["approved", "oof_precision", "rule_score", "global_train_matches"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return result


def apply_rules_to_test(
    test_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    confidence_floor: float,
) -> pd.DataFrame:
    test = test_df.copy()
    test["title_norm"] = _normalize_title(test["title"])

    direct_star, multi_claim = _extract_direct_star_signal(test["title_norm"])

    working = pd.DataFrame(
        {
            "id": test["id"],
            "title": test["title"],
            "rating": np.nan,
            "keyword": None,
            "confidence": 0.0,
            "rule_score": 0.0,
            "source": "none",
        }
    )

    direct_mask = direct_star.notna() & (~multi_claim)
    working.loc[direct_mask, "rating"] = direct_star[direct_mask]
    working.loc[direct_mask, "keyword"] = "<direct_star_phrase>"
    working.loc[direct_mask, "confidence"] = 1.0
    working.loc[direct_mask, "rule_score"] = 1e9
    working.loc[direct_mask, "source"] = "direct_star"

    approved = rules_df[rules_df["approved"] == 1].copy()
    approved = approved.sort_values(["rule_score", "oof_precision"], ascending=[False, False])

    for _, row in approved.iterrows():
        keyword = str(row["keyword"])
        rating = float(row["dominant_rating"])
        confidence = float(row["oof_precision"])
        score = float(row["rule_score"])

        if confidence < confidence_floor:
            continue

        regex = _build_keyword_regex(keyword)
        match = test["title_norm"].str.contains(regex, regex=True, na=False)

        better_rule = match & (
            working["source"].eq("none")
            | ((working["source"] == "keyword_pair") & (working["rule_score"] < score))
        )

        working.loc[better_rule, "rating"] = rating
        working.loc[better_rule, "keyword"] = keyword
        working.loc[better_rule, "confidence"] = confidence
        working.loc[better_rule, "rule_score"] = score
        working.loc[better_rule, "source"] = "keyword_pair"

    direct = working[["id", "rating", "source", "keyword", "confidence", "rule_score"]].copy()
    return direct


def merge_with_base_submission(
    direct_df: pd.DataFrame,
    base_submission_df: pd.DataFrame,
    min_merge_confidence: float,
) -> pd.DataFrame:
    required = {"id", "rating"}
    missing = required - set(base_submission_df.columns)
    if missing:
        raise ValueError(f"base submission missing columns: {sorted(missing)}")

    merged = base_submission_df[["id", "rating"]].merge(
        direct_df[["id", "rating", "source", "keyword", "confidence"]].rename(
            columns={"rating": "direct_rating"}
        ),
        on="id",
        how="left",
        validate="one_to_one",
    )

    merged["rating"] = pd.to_numeric(merged["rating"], errors="coerce")
    merged["direct_rating"] = pd.to_numeric(merged["direct_rating"], errors="coerce")

    apply_mask = (
        merged["direct_rating"].notna()
        & merged["source"].isin(["direct_star", "keyword_pair"])
        & (merged["confidence"] >= min_merge_confidence)
    )

    merged["final_rating"] = merged["rating"]
    merged.loc[apply_mask, "final_rating"] = merged.loc[apply_mask, "direct_rating"]
    merged["final_rating"] = merged["final_rating"].clip(1.0, 5.0)

    out = merged[["id", "final_rating"]].rename(columns={"final_rating": "rating"})
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and apply leakage-safe title keyword rules.")
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--keywords", nargs="*", default=None)
    parser.add_argument("--keywords-file", type=str, default="data/test_candidate_keywords.txt")
    parser.add_argument("--max-keywords", type=int, default=20)
    parser.add_argument("--min-keyword-len", type=int, default=3)

    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-global-matches", type=int, default=250)
    parser.add_argument("--min-oof-matches", type=int, default=120)
    parser.add_argument("--min-oof-precision", type=float, default=0.64)
    parser.add_argument("--min-margin-pct", type=float, default=14.0)
    parser.add_argument("--min-lift", type=float, default=1.12)
    parser.add_argument("--min-oof-consistency", type=float, default=0.80)
    parser.add_argument("--min-rule-confidence", type=float, default=0.66)

    parser.add_argument("--rules-out", type=str, default="data/direct_keyword_pairs.csv")
    parser.add_argument("--direct-out", type=str, default="data/direct_keyword_rate.csv")
    parser.add_argument("--rules-in", type=str, default=None, help="Reuse an existing rules CSV instead of mining from train.csv")
    parser.add_argument("--direct-in", type=str, default=None, help="Reuse an existing direct keyword-rate CSV for merge-only mode")

    parser.add_argument("--base-submission", type=str, default=None)
    parser.add_argument("--merged-out", type=str, default=None)
    parser.add_argument("--min-merge-confidence", type=float, default=0.70)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    direct_df: pd.DataFrame | None = None

    if args.direct_in:
        direct_path = _resolve_existing_path(args.direct_in, [], "direct keyword-rate csv")
        direct_df = pd.read_csv(direct_path)
        if not {"id", "rating"}.issubset(set(direct_df.columns)):
            raise ValueError("--direct-in CSV must contain columns: id, rating")
        print(f"Loaded direct keyword-rate CSV: {direct_path}")
    else:
        test_path = _resolve_existing_path(args.test_path, ["data/test.csv", "test.csv"], "test.csv")
        test_df = pd.read_csv(test_path)

        if args.rules_in:
            rules_path = _resolve_existing_path(args.rules_in, [], "rules csv")
            rules_df = pd.read_csv(rules_path)
            print(f"Loaded existing rules: {rules_path}")
        else:
            train_path = _resolve_existing_path(args.train_path, ["data/train.csv", "train.csv"], "train.csv")
            keywords = _load_keywords(args)
            print(f"Loaded {len(keywords)} candidate keywords")

            train_df = pd.read_csv(train_path)
            rules_df = mine_rules(train_df=train_df, test_df=test_df, keywords=keywords, args=args)

            rules_out = Path(args.rules_out)
            rules_out.parent.mkdir(parents=True, exist_ok=True)
            rules_df.to_csv(rules_out, index=False)
            approved = int((rules_df["approved"] == 1).sum())
            print(f"Saved rules: {rules_out.resolve()}")
            print(f"Approved rules: {approved}/{len(rules_df)}")

        direct_df = apply_rules_to_test(
            test_df=test_df,
            rules_df=rules_df,
            confidence_floor=args.min_rule_confidence,
        )
        direct_out = Path(args.direct_out)
        direct_out.parent.mkdir(parents=True, exist_ok=True)
        direct_df.to_csv(direct_out, index=False)

        direct_hits = int(direct_df["rating"].notna().sum())
        print(f"Saved direct keyword-rate predictions: {direct_out.resolve()}")
        print(f"Rows with direct override rating: {direct_hits}/{len(direct_df)}")

    if args.base_submission:
        base_path = _resolve_base_submission_path(args.base_submission)
        base_df = pd.read_csv(base_path)
        merged_df = merge_with_base_submission(
            direct_df=direct_df,
            base_submission_df=base_df,
            min_merge_confidence=args.min_merge_confidence,
        )
        if args.merged_out:
            merged_out = Path(args.merged_out)
        else:
            merged_out = Path("output") / "submissiona" / f"{base_path.stem}-keyword.csv"
        merged_out.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(merged_out, index=False)
        print(f"Saved merged submission: {merged_out.resolve()}")


if __name__ == "__main__":
    main()
