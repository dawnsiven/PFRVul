#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS=(
  "bigvul_cwe119_1_1"
  "bigvul_cwe125_1_1"
  "bigvul_cwe200_1_1"
  "bigvul_cwe264_1_1"
  "bigvul_cwe399_1_1"
)

for dataset in "${DATASETS[@]}"; do
  echo "========================================"
  echo "Processing dataset: ${dataset}"
  python3 improve_recall/merge_results.py --dataset "${dataset}"
  python3 improve_recall/summarize_results.py --dataset "${dataset}"
done

echo "========================================"
echo "All datasets processed successfully."
