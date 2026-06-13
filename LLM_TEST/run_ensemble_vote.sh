#!/bin/bash
set -euo pipefail

DATASET_TAG="${1:-bigvul_cwe119_1_1_CWE-119_0.5}"
PRESET="${2:-majority3}"
OUTPUT_DIR="${3:-LLM_TEST/output/ensemble_${DATASET_TAG}}"

DEEPSEEK_CSV="LLM_TEST/output/deepseek/${DATASET_TAG}/llm_predictions.csv"
BAIDU_CSV="LLM_TEST/output/baidu/${DATASET_TAG}/llm_predictions.csv"
LLAMA_CSV="LLM_TEST/output/llama3.2:3b/${DATASET_TAG}/llm_predictions.csv"
DSCODER_CSV="LLM_TEST/output/deepseek-coder:1.3b/${DATASET_TAG}/llm_predictions.csv"

ensure_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Missing file: ${path}"
        exit 1
    fi
}

run_vote() {
    python3 LLM_TEST/ensemble_vote.py "$@"
}

print_usage() {
    cat <<'EOF'
Usage:
  bash LLM_TEST/run_ensemble_vote.sh [dataset_tag] [preset] [output_dir]

Examples:
  bash LLM_TEST/run_ensemble_vote.sh
  bash LLM_TEST/run_ensemble_vote.sh bigvul_cwe119_1_1_CWE-119_0.5 any3
  bash LLM_TEST/run_ensemble_vote.sh bigvul_cwe119_1_1_CWE-119_0.5 threshold2of3
  bash LLM_TEST/run_ensemble_vote.sh bigvul_cwe119_1_1_CWE-119_0.5 weighted3_balanced

Presets:
  majority3
  any3
  threshold2of3
  all3
  weighted3_balanced
  weighted3_recall
  majority4
  any4
  threshold2of4
  threshold3of4
EOF
}

case "${PRESET}" in
    majority3|any3|threshold2of3|all3|weighted3_balanced|weighted3_recall)
        ensure_file "${DEEPSEEK_CSV}"
        ensure_file "${BAIDU_CSV}"
        ensure_file "${LLAMA_CSV}"
        COMMON_ARGS=(
            --inputs
            "deepseek=${DEEPSEEK_CSV}"
            "baidu=${BAIDU_CSV}"
            "llama=${LLAMA_CSV}"
            --intersection_only
            --output_dir "${OUTPUT_DIR}"
        )
        ;;
    majority4|any4|threshold2of4|threshold3of4)
        ensure_file "${DEEPSEEK_CSV}"
        ensure_file "${BAIDU_CSV}"
        ensure_file "${LLAMA_CSV}"
        ensure_file "${DSCODER_CSV}"
        COMMON_ARGS=(
            --inputs
            "deepseek=${DEEPSEEK_CSV}"
            "baidu=${BAIDU_CSV}"
            "llama=${LLAMA_CSV}"
            "dscoder=${DSCODER_CSV}"
            --intersection_only
            --output_dir "${OUTPUT_DIR}"
        )
        ;;
    help|-h|--help)
        print_usage
        exit 0
        ;;
    *)
        echo "Unknown preset: ${PRESET}"
        print_usage
        exit 1
        ;;
esac

mkdir -p "${OUTPUT_DIR}"

echo "Running ensemble vote"
echo "dataset_tag=${DATASET_TAG}"
echo "preset=${PRESET}"
echo "output_dir=${OUTPUT_DIR}"

case "${PRESET}" in
    majority3)
        run_vote "${COMMON_ARGS[@]}" --strategy majority --output_prefix majority_3model
        ;;
    any3)
        run_vote "${COMMON_ARGS[@]}" --strategy any --output_prefix any_3model
        ;;
    threshold2of3)
        run_vote "${COMMON_ARGS[@]}" --strategy threshold --threshold 2 --output_prefix threshold_2of3
        ;;
    all3)
        run_vote "${COMMON_ARGS[@]}" --strategy threshold --threshold 3 --output_prefix all_3model
        ;;
    weighted3_balanced)
        run_vote "${COMMON_ARGS[@]}" \
            --strategy weighted \
            --weights deepseek=1.0 baidu=1.3 llama=0.9 \
            --threshold 1.8 \
            --output_prefix weighted_3model_balanced
        ;;
    weighted3_recall)
        run_vote "${COMMON_ARGS[@]}" \
            --strategy weighted \
            --weights deepseek=1.1 baidu=0.7 llama=1.2 \
            --threshold 1.0 \
            --output_prefix weighted_3model_recall
        ;;
    majority4)
        run_vote "${COMMON_ARGS[@]}" --strategy majority --output_prefix majority_4model
        ;;
    any4)
        run_vote "${COMMON_ARGS[@]}" --strategy any --output_prefix any_4model
        ;;
    threshold2of4)
        run_vote "${COMMON_ARGS[@]}" --strategy threshold --threshold 2 --output_prefix threshold_2of4
        ;;
    threshold3of4)
        run_vote "${COMMON_ARGS[@]}" --strategy threshold --threshold 3 --output_prefix threshold_3of4
        ;;
esac

echo "Completed."
