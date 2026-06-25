#!/usr/bin/env python
"""Compatibility wrapper for direct title keyword extraction.

Delegates to code/models/keyword_direct_pipeline.py and writes the direct
keyword-rate CSV in the legacy default location.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy wrapper for title pickup")
    parser.add_argument("--test-path", type=str, default=None)
    parser.add_argument("--preview-out", type=str, default="data/previewSubmission.csv")
    parser.add_argument("--merge-semantic", type=str, default=None)
    parser.add_argument("--semantic-rating-col", type=str, default="rating")
    parser.add_argument("--merged-out", type=str, default="output/submissiona/submission_merged.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parents[2]
    target = root / "code" / "models" / "keyword_direct_pipeline.py"
    if not target.exists():
        raise FileNotFoundError(f"Missing pipeline: {target}")

    cmd = [sys.executable, str(target), "--direct-out", args.preview_out]

    if args.test_path:
        cmd.extend(["--test-path", args.test_path])
    if args.merge_semantic:
        cmd.extend(["--base-submission", args.merge_semantic, "--merged-out", args.merged_out])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
