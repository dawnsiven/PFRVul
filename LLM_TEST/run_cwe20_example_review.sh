#!/bin/bash
set -euo pipefail

print_usage() {
    cat <<'EOF'
Usage:
  bash LLM_TEST/run_cwe20_example_review.sh [limit]

Environment overrides:
  ENV_FILE
  CONFIG_FILE
  INPUT_JSON
  PROMPT_FILE
  EXAMPLE_RECORDS_JSON
  EXAMPLE_SIMILARITY_CSV
  WORKERS
  OUTPUT_ROOT
  OUTPUT_NAME
  RESULTS_CSV

Defaults:
  ENV_FILE=LLM_TEST/.env
  CONFIG_FILE=LLM_TEST/exp.yaml
  INPUT_JSON=LLM_TEST/intermediate/bigvul_cwe20_1_1_CodeBERT_imbalance_positive_0_1/positive_samples.jsonl
  PROMPT_FILE=LLM_TEST/Prompt/CWE-20_1.txt
  EXAMPLE_RECORDS_JSON=LLM_TEST/intermediate/bigvul_cwe20_1_1_train_example_bank.json
  EXAMPLE_SIMILARITY_CSV=
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage
    exit 0
fi

LIMIT="${1:-}"
ENV_FILE="${ENV_FILE:-LLM_TEST/.env}"
CONFIG_FILE="${CONFIG_FILE:-LLM_TEST/exp.yaml}"
INPUT_JSON="${INPUT_JSON:-LLM_TEST/intermediate/bigvul_cwe20_1_1_CodeBERT_imbalance_positive_0_1/positive_samples.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-LLM_TEST/Prompt/CWE-20_1.txt}"
EXAMPLE_RECORDS_JSON="${EXAMPLE_RECORDS_JSON:-LLM_TEST/intermediate/bigvul_cwe20_1_1_train_example_bank.json}"
EXAMPLE_SIMILARITY_CSV="${EXAMPLE_SIMILARITY_CSV:-}"
WORKERS="${WORKERS:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
OUTPUT_NAME="${OUTPUT_NAME:-}"
RESULTS_CSV="${RESULTS_CSV:-outputs/CodeBERT_imbalance/bigvul_cwe20_1_1/results.csv}"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Env file not found: ${ENV_FILE}"
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Config file not found: ${CONFIG_FILE}"
    exit 1
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "Prompt file not found: ${PROMPT_FILE}"
    exit 1
fi

if [[ ! -f "${EXAMPLE_RECORDS_JSON}" ]]; then
    echo "Example records json not found: ${EXAMPLE_RECORDS_JSON}"
    echo "Trying to build it from weighted train data ..."
    venv/bin/python LLM_TEST/build_train_example_bank.py \
        --train-json data/bigvul_cwe20_subsampled/alpaca/bigvul_cwe20_1_1_train_weighted.json \
        --output-json "${EXAMPLE_RECORDS_JSON}" \
        --min-weight 2.0
fi

if [[ ! -f "${EXAMPLE_RECORDS_JSON}" ]]; then
    echo "Failed to prepare example records json: ${EXAMPLE_RECORDS_JSON}"
    exit 1
fi

if [[ -n "${EXAMPLE_SIMILARITY_CSV}" && ! -f "${EXAMPLE_SIMILARITY_CSV}" ]]; then
    echo "Example similarity csv not found: ${EXAMPLE_SIMILARITY_CSV}"
    exit 1
fi

if [[ ! -f "${INPUT_JSON}" ]]; then
    echo "Input JSON not found: ${INPUT_JSON}"
    echo "Trying to generate it from ${RESULTS_CSV} ..."
    venv/bin/python LLM_TEST/extract_positive_samples.py \
        --config "${CONFIG_FILE}" \
        --env_file "${ENV_FILE}" \
        --results_csv "${RESULTS_CSV}" \
        --min_prob 0 \
        --max_prob 1 \
        --output_subdir "bigvul_cwe20_1_1_CodeBERT_imbalance_positive_0_1"
fi

if [[ ! -f "${INPUT_JSON}" ]]; then
    echo "Failed to prepare input json: ${INPUT_JSON}"
    exit 1
fi

echo "Starting CWE-20 example-based review"
echo "config=${CONFIG_FILE}"
echo "env=${ENV_FILE}"
echo "input_json=${INPUT_JSON}"
echo "prompt_file=${PROMPT_FILE}"
echo "example_records_json=${EXAMPLE_RECORDS_JSON}"
echo "example_similarity_csv=${EXAMPLE_SIMILARITY_CSV:-none}"
echo "workers=${WORKERS}"
echo "limit=${LIMIT:-all}"

set -- \
    venv/bin/python \
    LLM_TEST/llm_api_judge_with_examples.py \
    --config "${CONFIG_FILE}" \
    --env_file "${ENV_FILE}" \
    --input_json "${INPUT_JSON}" \
    --prompt_file "${PROMPT_FILE}" \
    --example_records_json "${EXAMPLE_RECORDS_JSON}" \
    --workers "${WORKERS}" \
    --output_by_prompt_version

if [[ -n "${EXAMPLE_SIMILARITY_CSV}" ]]; then
    set -- "$@" --example_similarity_csv "${EXAMPLE_SIMILARITY_CSV}"
fi

if [[ -n "${LIMIT}" ]]; then
    set -- "$@" --limit "${LIMIT}"
fi

if [[ -n "${OUTPUT_ROOT}" ]]; then
    set -- "$@" --output_root "${OUTPUT_ROOT}"
fi

if [[ -n "${OUTPUT_NAME}" ]]; then
    set -- "$@" --output_name "${OUTPUT_NAME}"
fi

command "$@"
printf '%s\n' "Completed."
