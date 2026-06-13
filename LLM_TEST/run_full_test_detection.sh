#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
    cat <<'EOF'
Usage:
  bash LLM_TEST/run_full_test_detection.sh <data_json> <prompt_file> <output_name> [results_csv] [limit] [workers]

Example:
  bash LLM_TEST/run_full_test_detection.sh \
    data/bigvul_subsampled/alpaca/bigvul_0-512_5_test.json \
    LLM_TEST/Prompt/CWE-119_0.5.txt \
    fulltest_bigvul_0-512_5_deepseek_cwe119 \
    "" \
    100 \
    4
EOF
    exit 1
fi

DATA_JSON="$1"
PROMPT_FILE="$2"
OUTPUT_NAME="$3"
RESULTS_CSV="${4:-}"
LIMIT="${5:-}"
WORKERS="${6:-1}"

PREP_SUBDIR="${OUTPUT_NAME}_prepared"
INPUT_JSON="LLM_TEST/intermediate/${PREP_SUBDIR}/full_test_samples.jsonl"
OUTPUT_DIR="LLM_TEST/output/${OUTPUT_NAME}"

PREPARE_CMD=(
    python3 LLM_TEST/prepare_full_test_samples.py
    --data_json "$DATA_JSON"
    --output_subdir "$PREP_SUBDIR"
)

if [[ -n "$RESULTS_CSV" ]]; then
    PREPARE_CMD+=(--results_csv "$RESULTS_CSV")
fi

if [[ -n "$LIMIT" ]]; then
    PREPARE_CMD+=(--limit "$LIMIT")
fi

"${PREPARE_CMD[@]}"

JUDGE_CMD=(
    python3 LLM_TEST/llm_api_judge.py
    --input_json "$INPUT_JSON"
    --prompt_file "$PROMPT_FILE"
    --output_name "$OUTPUT_NAME"
    --workers "$WORKERS"
)

"${JUDGE_CMD[@]}"

python3 LLM_TEST/evaluate_llm_full_dataset.py \
    --llm_predictions_csv "${OUTPUT_DIR}/llm_predictions.csv" \
    --output_dir "$OUTPUT_DIR"

echo "prepared_input_json=${INPUT_JSON}"
echo "output_dir=${OUTPUT_DIR}"
echo "metrics_json=${OUTPUT_DIR}/full_llm_metrics.json"
