#!/bin/bash
set -euo pipefail

DATASET_NAME=${1:-}
RESULT_MODEL_NAME=${2:-}
LLM_MODEL_NAME=${3:-}
LENGTH=${4:-}
POS_RATIO=""
BATCH_SIZE=""
LENGTH_BUCKET="0-512"
CUDA="0"
OOF_RUN_TAG=""

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

if [ $# -lt 5 ]; then
    echo "Usage (imbalance): $0 <DATASET_NAME> <RESULT_MODEL_NAME> <LLM_MODEL_NAME> <LENGTH> <POS_RATIO> <BATCH_SIZE> [LENGTH_BUCKET] [CUDA] [OOF_RUN_TAG]"
    echo "Usage (non-imbalance): $0 <DATASET_NAME> <RESULT_MODEL_NAME> <LLM_MODEL_NAME> <LENGTH> <BATCH_SIZE> [LENGTH_BUCKET] [CUDA] [OOF_RUN_TAG]"
    echo "Env:"
    echo "  FORCE_SMALL_MODEL_TRAIN=1    Re-run small-model training even if checkpoint exists"
    echo "  FORCE_REVIEWER_EXPORT=1      Re-run reviewer CSV export even if reviewer CSVs exist"
    echo "  FORCE_OOF=1                  Re-run OOF generation even if reviewer_train.csv exists"
    echo "  FORCE_PREP=1                 Re-run reviewer JSON preparation even if rebucketed data exists"
    echo "  FORCE_FINETUNE=1             Re-run reviewer LoRA fine-tuning even if checkpoint exists"
    echo "  FORCE_INFERENCE=1            Re-run reviewer inference and merge even if merged_metrics.json exists"
    exit 1
fi

# Parsing rules:
# - non-imbalance:
#   $5 = BATCH_SIZE (numeric)
#   $6 = optional LENGTH_BUCKET
#   $7 = optional CUDA
#   $8 = optional OOF_RUN_TAG
# - imbalance:
#   $5 = POS_RATIO (numeric)
#   $6 = BATCH_SIZE (numeric)
#   $7 = optional LENGTH_BUCKET
#   $8 = optional CUDA
#   $9 = optional OOF_RUN_TAG
if [ $# -ge 6 ] && [[ "$5" =~ ^[0-9]+$ ]] && [[ "$6" =~ ^[0-9]+$ ]]; then
    POS_RATIO="$5"
    BATCH_SIZE="$6"
    LENGTH_BUCKET=${7:-"0-512"}
    CUDA=${8:-"0"}
    OOF_RUN_TAG=${9:-""}
else
    BATCH_SIZE="$5"
    LENGTH_BUCKET=${6:-"0-512"}
    CUDA=${7:-"0"}
    OOF_RUN_TAG=${8:-""}
fi

if [[ ${BASH_VERSINFO[0]} -lt 4 ]]; then
    echo "Bash version 4.0 or higher is required."
    exit 1
fi

FORCE_SMALL_MODEL_TRAIN=${FORCE_SMALL_MODEL_TRAIN:-0}
FORCE_REVIEWER_EXPORT=${FORCE_REVIEWER_EXPORT:-0}
FORCE_OOF=${FORCE_OOF:-0}
FORCE_PREP=${FORCE_PREP:-0}
FORCE_FINETUNE=${FORCE_FINETUNE:-0}
FORCE_INFERENCE=${FORCE_INFERENCE:-0}

DATASET_TAG="${DATASET_NAME}_${LENGTH}"
SMALL_MODEL_OUTPUT_DIR="${REPO_ROOT}/outputs/${RESULT_MODEL_NAME}/${DATASET_TAG}"
REVIEWER_DATA_DIR="${REPO_ROOT}/reviewer_finetune_data/${RESULT_MODEL_NAME}/${DATASET_TAG}_length_rebucketed"
LORA_OUTPUT_DIR="${REPO_ROOT}/outputs/${LLM_MODEL_NAME}_lora/${RESULT_MODEL_NAME}_${DATASET_TAG}_${LENGTH_BUCKET}"
OOF_OUTPUT_ROOT="${REPO_ROOT}/outputs/${RESULT_MODEL_NAME}_oof/${DATASET_TAG}"

if [[ -n "${POS_RATIO}" ]]; then
    DATASET_TAG="${DATASET_NAME}_${LENGTH}_${POS_RATIO}"
    SMALL_MODEL_OUTPUT_DIR="${REPO_ROOT}/outputs/${RESULT_MODEL_NAME}_imbalance/${DATASET_TAG}"
    REVIEWER_DATA_DIR="${REPO_ROOT}/reviewer_finetune_data/${RESULT_MODEL_NAME}_imbalance/${DATASET_TAG}_length_rebucketed"
    LORA_OUTPUT_DIR="${REPO_ROOT}/outputs/${LLM_MODEL_NAME}_lora/${RESULT_MODEL_NAME}_imbalance_${DATASET_TAG}_${LENGTH_BUCKET}"
    OOF_OUTPUT_ROOT="${REPO_ROOT}/outputs/${RESULT_MODEL_NAME}_imbalance_oof/${DATASET_TAG}"
fi

if [[ -z "${OOF_RUN_TAG}" ]]; then
    OOF_RUN_TAG="pipeline_default"
fi

OOF_REVIEWER_TRAIN_CSV="${OOF_OUTPUT_ROOT}/${OOF_RUN_TAG}/reviewer_train.csv"
OOF_SUMMARY_JSON="${OOF_OUTPUT_ROOT}/${OOF_RUN_TAG}/oof_summary.json"
OOF_OUTPUT_DIR="${OOF_OUTPUT_ROOT}/${OOF_RUN_TAG}"

echo "Dataset: ${DATASET_NAME}"
echo "Result model: ${RESULT_MODEL_NAME}"
echo "Reviewer model: ${LLM_MODEL_NAME}"
echo "Length: ${LENGTH}"
if [[ -n "${POS_RATIO}" ]]; then
    echo "POS_RATIO: ${POS_RATIO}"
else
    echo "POS_RATIO: <non-imbalance mode>"
fi
echo "Batch size: ${BATCH_SIZE}"
echo "Length bucket: ${LENGTH_BUCKET}"
echo "CUDA: ${CUDA}"
echo "OOF run tag: ${OOF_RUN_TAG}"

small_model_checkpoint_exists() {
    [[ -f "${SMALL_MODEL_OUTPUT_DIR}/checkpoint-best-f1/model.bin" ]]
}

reviewer_csvs_exist() {
    [[ -f "${SMALL_MODEL_OUTPUT_DIR}/reviewer_train.csv" ]] && \
    [[ -f "${SMALL_MODEL_OUTPUT_DIR}/reviewer_val.csv" ]] && \
    [[ -f "${SMALL_MODEL_OUTPUT_DIR}/reviewer_test.csv" ]]
}

oof_exists() {
    [[ -f "${OOF_REVIEWER_TRAIN_CSV}" ]] && [[ -f "${OOF_SUMMARY_JSON}" ]]
}

rebucketed_data_exists() {
    [[ -f "${REVIEWER_DATA_DIR}/train_${LENGTH_BUCKET}.json" ]] && \
    [[ -f "${REVIEWER_DATA_DIR}/val_${LENGTH_BUCKET}.json" ]] && \
    [[ -f "${REVIEWER_DATA_DIR}/test_${LENGTH_BUCKET}.json" ]]
}

lora_checkpoint_exists() {
    find "${LORA_OUTPUT_DIR}" -maxdepth 1 -type d -name 'epoch-*' | grep -q .
}

merged_results_exist() {
    [[ -f "${LORA_OUTPUT_DIR}/merged_metrics.json" ]]
}

run_small_model_train() {
    if small_model_checkpoint_exists && [[ "${FORCE_SMALL_MODEL_TRAIN}" != "1" ]]; then
        echo "[Skip] Small-model checkpoint already exists: ${SMALL_MODEL_OUTPUT_DIR}/checkpoint-best-f1/model.bin"
        return
    fi

    echo "[Run] Small-model training"
    if [[ -n "${POS_RATIO}" ]]; then
        "${REPO_ROOT}/scripts/train_imbalance.sh" "${DATASET_NAME}" "${RESULT_MODEL_NAME}" "${LENGTH}" "${POS_RATIO}" "${CUDA}"
    else
        "${REPO_ROOT}/scripts/train.sh" "${DATASET_NAME}" "${RESULT_MODEL_NAME}" "${LENGTH}" "${CUDA}"
    fi
}

run_reviewer_export() {
    if reviewer_csvs_exist && [[ "${FORCE_REVIEWER_EXPORT}" != "1" ]]; then
        echo "[Skip] Reviewer CSVs already exist under: ${SMALL_MODEL_OUTPUT_DIR}"
        return
    fi

    echo "[Run] Export original reviewer CSVs"
    if [[ -n "${POS_RATIO}" ]]; then
        "${REPO_ROOT}/scripts/test_imbalance_test.sh" "${DATASET_NAME}" "${RESULT_MODEL_NAME}" "${LENGTH}" "${POS_RATIO}" "${CUDA}"
    else
        "${REPO_ROOT}/scripts/test_imbalance_test.sh" "${DATASET_NAME}" "${RESULT_MODEL_NAME}" "${LENGTH}" "${CUDA}"
    fi
}

run_oof() {
    if oof_exists && [[ "${FORCE_OOF}" != "1" ]]; then
        echo "[Skip] OOF reviewer_train.csv already exists: ${OOF_REVIEWER_TRAIN_CSV}"
        return
    fi

    if [[ -d "${OOF_OUTPUT_DIR}" ]] && [[ -n "$(find "${OOF_OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]] && [[ "${FORCE_OOF}" != "1" ]]; then
        echo "[Stop] OOF output directory already exists but is incomplete:"
        echo "  ${OOF_OUTPUT_DIR}"
        echo "Missing required files:"
        [[ -f "${OOF_REVIEWER_TRAIN_CSV}" ]] || echo "  ${OOF_REVIEWER_TRAIN_CSV}"
        [[ -f "${OOF_SUMMARY_JSON}" ]] || echo "  ${OOF_SUMMARY_JSON}"
        echo "Use one of the following options:"
        echo "  1. set a different OOF_RUN_TAG"
        echo "  2. remove the incomplete directory"
        echo "  3. rerun with FORCE_OOF=1 after cleanup"
        exit 1
    fi

    echo "[Run] Generate OOF reviewer_train.csv"
    if [[ -n "${POS_RATIO}" ]]; then
        OOF_RUN_TAG="${OOF_RUN_TAG}" "${REPO_ROOT}/scripts/train_imbalance_oof.sh" \
            "${DATASET_NAME}" "${RESULT_MODEL_NAME}" "${LENGTH}" "${POS_RATIO}" "${CUDA}"
    else
        OOF_RUN_TAG="${OOF_RUN_TAG}" "${REPO_ROOT}/scripts/train_imbalance_oof.sh" \
            "${DATASET_NAME}" "${RESULT_MODEL_NAME}" "${LENGTH}" "${CUDA}"
    fi
}

run_prepare_reviewer_data() {
    if rebucketed_data_exists && [[ "${FORCE_PREP}" != "1" ]]; then
        echo "[Skip] Rebucketed reviewer data already exists under: ${REVIEWER_DATA_DIR}"
        return
    fi

    echo "[Run] Prepare reviewer JSON data with OOF train"
    if [[ -n "${POS_RATIO}" ]]; then
        "${REPO_ROOT}/data_process/rebucket_reviewer_data_oof_train.sh" \
            "${DATASET_NAME}" "${RESULT_MODEL_NAME}" "${LENGTH}" "${POS_RATIO}" "${OOF_RUN_TAG}"
    else
        "${REPO_ROOT}/data_process/rebucket_reviewer_data_oof_train.sh" \
            "${DATASET_NAME}" "${RESULT_MODEL_NAME}" "${LENGTH}" "${OOF_RUN_TAG}"
    fi
}

run_lora_finetune() {
    if lora_checkpoint_exists && [[ "${FORCE_FINETUNE}" != "1" ]]; then
        echo "[Skip] Reviewer LoRA checkpoint already exists under: ${LORA_OUTPUT_DIR}"
        return
    fi

    echo "[Run] Reviewer LoRA fine-tuning"
    if [[ -n "${POS_RATIO}" ]]; then
        "${REPO_ROOT}/scripts/finetune_test.sh" \
            "${DATASET_NAME}" "${RESULT_MODEL_NAME}" "${LLM_MODEL_NAME}" "${LENGTH}" \
            "${POS_RATIO}" "${BATCH_SIZE}" "${LENGTH_BUCKET}" "${CUDA}"
    else
        "${REPO_ROOT}/scripts/finetune_test.sh" \
            "${DATASET_NAME}" "${RESULT_MODEL_NAME}" "${LLM_MODEL_NAME}" "${LENGTH}" \
            "${BATCH_SIZE}" "${LENGTH_BUCKET}" "${CUDA}"
    fi
}

run_lora_inference_and_merge() {
    if merged_results_exist && [[ "${FORCE_INFERENCE}" != "1" ]]; then
        echo "[Skip] Merged reviewer outputs already exist under: ${LORA_OUTPUT_DIR}"
        return
    fi

    echo "[Run] Reviewer inference and final merge"
    if [[ -n "${POS_RATIO}" ]]; then
        "${REPO_ROOT}/scripts/inference_finetune_test.sh" \
            "${DATASET_NAME}" "${RESULT_MODEL_NAME}" "${LLM_MODEL_NAME}" "${LENGTH}" \
            "${POS_RATIO}" "${LENGTH_BUCKET}" "${CUDA}"
    else
        "${REPO_ROOT}/scripts/inference_finetune_test.sh" \
            "${DATASET_NAME}" "${RESULT_MODEL_NAME}" "${LLM_MODEL_NAME}" "${LENGTH}" \
            "${LENGTH_BUCKET}" "${CUDA}"
    fi
}

run_small_model_train
run_reviewer_export
run_oof
run_prepare_reviewer_data
run_lora_finetune
run_lora_inference_and_merge

echo
echo "Pipeline completed."
echo "Small-model results dir: ${SMALL_MODEL_OUTPUT_DIR}"
echo "OOF reviewer train CSV: ${OOF_REVIEWER_TRAIN_CSV}"
echo "Rebucketed reviewer data dir: ${REVIEWER_DATA_DIR}"
echo "Reviewer LoRA output dir: ${LORA_OUTPUT_DIR}"
