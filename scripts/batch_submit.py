#!/usr/bin/env python3
"""
Re-authenticate with Kaggle API token and batch submit 9 predictions for 2026-06-22.
"""

import os
import sys
import time
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

TOKENS = [
    "KGAT_9895fa87525d5a9a3514ae8bd156320b",
    "KGAT_95032a984dab4b2545f71383d9913c63",
    "KGAT_625ace77411f574aea39ba8c22bccbe9",
]

COMPETITION = "comp-5434-2526-sem-3-project"

SUBMISSIONS = [
    ("output/archive/submissions/sub-20260622-01-ve65-rlg35.csv", "VE65%+rlg35%"),
    ("output/archive/submissions/sub-20260622-02-ve70-rlg30.csv", "VE70%+rlg30%"),
    ("output/archive/submissions/sub-20260622-03-ve75-rlg25.csv", "VE75%+rlg25%"),
    ("output/archive/submissions/sub-20260622-05-base40-large30-rlg30.csv", "base40+large30+rlg30%"),
    ("output/archive/submissions/sub-20260622-06-base30-large40-rlg30.csv", "base30+large40+rlg30%"),
    ("output/archive/submissions/sub-20260622-07-ve55-multi-stack.csv", "VE55%+multi-stack"),
    ("output/archive/submissions/sub-20260622-08-ve50-all-stack.csv", "VE50%+all-stack"),
    ("output/archive/submissions/sub-20260622-09-geometric.csv", "geometric-mean"),
    ("output/archive/submissions/sub-20260622-10-harmonic.csv", "harmonic-mean"),
]


def try_auth_with_token(token):
    """Write token as kaggle.json API key, then authenticate."""
    creds = {"username": "rickyma1028", "key": token}
    creds_path = os.path.expanduser("~/.kaggle/kaggle.json")
    os.makedirs(os.path.dirname(creds_path), exist_ok=True)
    with open(creds_path, "w") as f:
        json.dump(creds, f, indent=2)

    # Also write as access_token for CLI
    token_path = os.path.expanduser("~/.kaggle/access_token")
    with open(token_path, "w") as f:
        f.write(token)

    # Back up and remove OAuth credentials to avoid conflict
    old_creds = os.path.expanduser("~/.kaggle/credentials.json")
    if os.path.exists(old_creds):
        backup = old_creds + ".bak"
        if not os.path.exists(backup):
            os.rename(old_creds, backup)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    # Test with a lightweight call
    subs = api.competition_submissions(COMPETITION)
    print(f"  Token {token[:16]}... works (found {len(subs)} submissions)")
    return api


def main():
    api = None
    for token in TOKENS:
        print(f"Trying auth with token...")
        try:
            api = try_auth_with_token(token)
            break
        except Exception as e:
            print(f"  Failed: {type(e).__name__}: {e}")

    if api is None:
        print("ERROR: All tokens failed")
        sys.exit(1)

    print(f"\nAuthenticated. Submitting {len(SUBMISSIONS)} files to {COMPETITION}\n")

    results = []
    for filepath, message in SUBMISSIONS:
        fname = Path(filepath).name
        full_path = str(Path(filepath).resolve())
        print(f"Submitting: {fname} ... ", end="", flush=True)
        try:
            start = time.time()
            api.competition_submit(full_path, message, COMPETITION)
            elapsed = time.time() - start
            print(f"OK ({elapsed:.1f}s)")
            results.append((fname, "SUCCESS", ""))
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            results.append((fname, "FAILED", str(e)))
        time.sleep(2)

    print("\n" + "=" * 60)
    print("SUBMISSION SUMMARY")
    print("=" * 60)
    success = 0
    for fname, status, err in results:
        icon = "OK" if status == "SUCCESS" else "FAIL"
        extra = f" ({err})" if err else ""
        print(f"  [{icon}] {fname}{extra}")
        if status == "SUCCESS":
            success += 1
    print(f"\n{success}/{len(results)} submitted successfully.")


if __name__ == "__main__":
    main()
