#!/usr/bin/env python
"""Compatibility wrapper for the robust keyword direct-pair miner.

This wrapper delegates to code/models/keyword_direct_pipeline.py, which is the
official keyword pipeline.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for keyword rule finding.")
    parser.add_argument("--keywords", nargs="*", default=None)
    parser.add_argument("--keywords-file", type=str, default="data/test_candidate_keywords.txt")
    parser.add_argument("--max-keywords", type=int, default=50)
    parser.add_argument("--train-path", type=str, default=None)
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--out-csv", type=str, default="data/keyword_rules.csv")

    parser.add_argument("--min-train-matches", type=int, default=250)
    parser.add_argument("--min-dominant-pct", type=float, default=65.0)
    parser.add_argument("--min-margin-pct", type=float, default=14.0)
    parser.add_argument("--min-lift", type=float, default=1.12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[2]
    target = root / "code" / "models" / "keyword_direct_pipeline.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing pipeline: {target}")

    cmd = [
        sys.executable,
        str(target),
        "--rules-out",
        args.out_csv,
        "--keywords-file",
        args.keywords_file,
        "--max-keywords",
        str(args.max_keywords),
        "--min-global-matches",
        str(args.min_train_matches),
        "--min-margin-pct",
        str(args.min_margin_pct),
        "--min-lift",
        str(args.min_lift),
    ]

    if args.train_path:
        cmd.extend(["--train-path", args.train_path])
    if args.test_path:
        cmd.extend(["--test-path", args.test_path])
    if args.keywords:
        cmd.extend(["--keywords", *args.keywords])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
