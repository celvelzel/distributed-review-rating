#!/usr/bin/env python
"""Compatibility wrapper for merging direct keyword-rate predictions.

Delegates to code/models/keyword_direct_pipeline.py using an existing
advanced submission CSV as base.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge direct keyword rate CSV with advanced submission CSV.")
    parser.add_argument("--preview-csv", type=str, required=True, help="Ignored (kept for backward compatibility).")
    parser.add_argument("--semantic-csv", type=str, required=True, help="Advanced model submission CSV.")
    parser.add_argument("--official-out", type=str, default="data/submission.csv")
    parser.add_argument("--fill-empty-only", action="store_true", help="Ignored: robust pipeline always does controlled override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[1]
    target = root / "code" / "models" / "keyword_direct_pipeline.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing pipeline: {target}")

    cmd = [
        sys.executable,
        str(target),
        "--base-submission",
        args.semantic_csv,
        "--merged-out",
        args.official_out,
        "--direct-out",
        args.preview_csv,
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
