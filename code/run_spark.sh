#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'Usage: bash code/run_spark.sh python code/etl/run_etl.py [args...]\n' >&2
  exit 1
fi

launcher="$1"
shift

if [ "$launcher" = "python" ]; then
  exec spark-submit \
    --master local[*] \
    --driver-memory 4g \
    "$@"
fi

exec "$launcher" \
  --master local[*] \
  --driver-memory 4g \
  "$@"
