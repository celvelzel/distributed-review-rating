#!/usr/bin/env bash
# Authenticate to Kaggle by trying tokens from config/kaggle_tokens.json
# Usage: bash scripts/kaggle_auth.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG="$PROJECT_ROOT/config/kaggle_tokens.json"

if [ ! -f "$CONFIG" ]; then
  echo "ERROR: $CONFIG not found" >&2
  exit 1
fi

# Parse tokens using Python (available on all HPC systems)
PYTHON_CMD=$(command -v python3 || command -v python)
TOKENS=$($PYTHON_CMD -c "
import json, sys
with open('$CONFIG') as f:
    data = json.load(f)
for t in data['tokens']:
    print(t)
")

for TOKEN in $TOKENS; do
  echo "Trying token: ${TOKEN:0:12}..."
  export KAGGLE_API_TOKEN="$TOKEN"
  if kaggle competitions list -s "comp-5434-2526-sem-3-project" > /dev/null 2>&1; then
    echo "Token works: ${TOKEN:0:12}..."
    # Save to standard location
    mkdir -p ~/.kaggle
    echo "$TOKEN" > ~/.kaggle/access_token
    chmod 600 ~/.kaggle/access_token
    echo "Saved to ~/.kaggle/access_token"
    exit 0
  fi
  echo "  Failed."
done

echo "ERROR: All tokens failed" >&2
exit 1
