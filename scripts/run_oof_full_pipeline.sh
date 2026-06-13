#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
PIPELINE_SCRIPT="${REPO_ROOT}/scripts/run_reviewer_oof_pipeline.sh"

if [[ ! -x "${PIPELINE_SCRIPT}" ]]; then
    echo "Pipeline script is missing or not executable:"
    echo "  ${PIPELINE_SCRIPT}"
    exit 1
fi

if [[ $# -lt 5 ]]; then
    echo "OOF multi-dataset pipeline wrapper"
    echo
    echo "This script runs the complete OOF reviewer workflow by reusing:"
    echo "  ${PIPELINE_SCRIPT}"
    echo
    echo "It supports one dataset or multiple datasets separated by commas."
    echo
    echo "Examples:"
    echo "  $0 cvefixes_cwe352 CodeBERT llama3.2 0-512 4"
    echo "  $0 cvefixes_cwe352,cvefixes_cwe79 CodeBERT llama3.2 0-512 4"
    echo "  $0 bigvul_cwe20,bigvul_cwe79 CodeBERT llama3.2 1 1 4 0-512 0 batch_run"
    echo
    echo "Workflow steps:"
    echo "  0. small-model train"
    echo "  1. export original reviewer CSVs"
    echo "  2. generate OOF reviewer_train.csv"
    echo "  3. build reviewer finetune JSON with OOF train"
    echo "  4. reviewer LoRA finetune"
    echo "  5. reviewer inference + final merge"
    echo
    echo "Usage (imbalance): $0 <DATASET_NAME[,DATASET_NAME...]> <RESULT_MODEL_NAME> <LLM_MODEL_NAME> <LENGTH> <POS_RATIO> <BATCH_SIZE> [LENGTH_BUCKET] [CUDA] [OOF_RUN_TAG]"
    echo "Usage (non-imbalance): $0 <DATASET_NAME[,DATASET_NAME...]> <RESULT_MODEL_NAME> <LLM_MODEL_NAME> <LENGTH> <BATCH_SIZE> [LENGTH_BUCKET] [CUDA] [OOF_RUN_TAG]"
    echo
    echo "Useful env vars:"
    echo "  OOF_FOLDS=5"
    echo "  OOF_SEED=42"
    echo "  FORCE_SMALL_MODEL_TRAIN=1"
    echo "  FORCE_REVIEWER_EXPORT=1"
    echo "  FORCE_OOF=1"
    echo "  FORCE_PREP=1"
    echo "  FORCE_FINETUNE=1"
    echo "  FORCE_INFERENCE=1"
    exit 1
fi

DATASET_LIST_RAW="$1"
shift

IFS=',' read -r -a DATASETS <<< "${DATASET_LIST_RAW}"

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "No dataset names were provided."
    exit 1
fi

for i in "${!DATASETS[@]}"; do
    DATASET_NAME="${DATASETS[$i]}"
    DATASET_NAME="${DATASET_NAME#"${DATASET_NAME%%[![:space:]]*}"}"
    DATASET_NAME="${DATASET_NAME%"${DATASET_NAME##*[![:space:]]}"}"

    if [[ -z "${DATASET_NAME}" ]]; then
        echo "Encountered an empty dataset name in: ${DATASET_LIST_RAW}"
        exit 1
    fi

    echo "============================================================"
    echo "[$((i + 1))/${#DATASETS[@]}] Running OOF pipeline for dataset: ${DATASET_NAME}"
    echo "============================================================"

    "${PIPELINE_SCRIPT}" "${DATASET_NAME}" "$@"

    echo
    echo "[Done] ${DATASET_NAME}"
    echo
done
