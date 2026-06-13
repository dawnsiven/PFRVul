#!/bin/bash
set -euo pipefail

DATASET_NAME=${1:-}
RESULT_MODEL_NAME=${2:-}
MODEL_NAME=${3:-}
LENGTH=${4:-}
POS_RATIO=""
BATCH_SIZE=""
LENGTH_BUCKET="0-512"
CUDA="0"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# Check if the required parameters are provided
if [ $# -lt 5 ]; then
    echo "Usage (imbalance): $0 <DATASET_NAME> <RESULT_MODEL_NAME> <LLM_MODEL_NAME> <LENGTH> <POS_RATIO> <BATCH_SIZE> [LENGTH_BUCKET] [CUDA]"
    echo "Usage (non-imbalance): $0 <DATASET_NAME> <RESULT_MODEL_NAME> <LLM_MODEL_NAME> <LENGTH> <BATCH_SIZE> [LENGTH_BUCKET] [CUDA]"
    exit 1
fi

# Parsing rules:
# - non-imbalance:
#   $5 = BATCH_SIZE (numeric)
#   $6 = optional LENGTH_BUCKET (usually contains '-')
# - imbalance:
#   $5 = POS_RATIO (numeric)
#   $6 = BATCH_SIZE (numeric)
# We distinguish them mainly by whether $6 is numeric.
if [ $# -ge 6 ] && [[ "$5" =~ ^[0-9]+$ ]] && [[ "$6" =~ ^[0-9]+$ ]]; then
    POS_RATIO="$5"
    BATCH_SIZE="$6"
    LENGTH_BUCKET=${7:-"0-512"}
    CUDA=${8:-"0"}
else
    BATCH_SIZE="$5"
    LENGTH_BUCKET=${6:-"0-512"}
    CUDA=${7:-"0"}
fi

if [[ ${BASH_VERSINFO[0]} -lt 4 ]]; then
    echo "Bash version 4.0 or higher is required."
    exit 1
fi

declare -A MODEL_MAP
MODEL_MAP["llama3"]='meta-llama/Meta-Llama-3-8B'
MODEL_MAP["llama3.1"]='meta-llama/Llama-3.1-8B'
MODEL_MAP["llama2"]="meta-llama/Llama-2-7b-hf"
MODEL_MAP["codellama"]="codellama/CodeLlama-7b-hf"
MODEL_MAP["llama3.2"]="meta-llama/Llama-3.2-1B"

if [[ -z "${MODEL_MAP[$MODEL_NAME]}" ]]; then
    echo "Unknown LLM model alias: ${MODEL_NAME}"
    echo "Available aliases: ${!MODEL_MAP[*]}"
    exit 1
fi

REVIEWER_DATA_DIR="reviewer_finetune_data"
DATASET_TAG="${DATASET_NAME}_${LENGTH}"
REVIEWER_DATA_SUBDIR="${REVIEWER_DATA_DIR}/${RESULT_MODEL_NAME}/${DATASET_TAG}"
REBUCKETED_DATA_SUBDIR="${REVIEWER_DATA_DIR}/${RESULT_MODEL_NAME}/${DATASET_TAG}_length_rebucketed"
OUTPUT_STEM="${RESULT_MODEL_NAME}_${DATASET_TAG}_${LENGTH_BUCKET}"

if [[ -n "${POS_RATIO}" ]]; then
    DATASET_TAG="${DATASET_NAME}_${LENGTH}_${POS_RATIO}"
    REVIEWER_DATA_SUBDIR="${REVIEWER_DATA_DIR}/${RESULT_MODEL_NAME}_imbalance/${DATASET_TAG}"
    REBUCKETED_DATA_SUBDIR="${REVIEWER_DATA_DIR}/${RESULT_MODEL_NAME}_imbalance/${DATASET_TAG}_length_rebucketed"
    OUTPUT_STEM="${RESULT_MODEL_NAME}_imbalance_${DATASET_TAG}_${LENGTH_BUCKET}"
fi

TRAIN_JSON="${REBUCKETED_DATA_SUBDIR}/train_${LENGTH_BUCKET}.json"
VAL_JSON="${REBUCKETED_DATA_SUBDIR}/val_${LENGTH_BUCKET}.json"
TEST_JSON="${REBUCKETED_DATA_SUBDIR}/test_${LENGTH_BUCKET}.json"
OUTPUT_DIR="outputs/${MODEL_NAME}_lora/${OUTPUT_STEM}/"

CONTEXT_LENGTH="512"
if [[ "${LENGTH}" == *-* ]]; then
    CONTEXT_LENGTH=$(echo "${LENGTH}" | awk -F'-' '{print $2}')
fi

mkdir -p "${OUTPUT_DIR}"

echo "Batch size: ${BATCH_SIZE}"
echo "Length: ${LENGTH}"
echo "Context length: ${CONTEXT_LENGTH}"
if [[ -n "${POS_RATIO}" ]]; then
    echo "POS_RATIO: ${POS_RATIO}"
else
    echo "POS_RATIO: <non-imbalance mode>"
fi
echo "Reviewer result model: ${RESULT_MODEL_NAME}"
echo "Length bucket: ${LENGTH_BUCKET}"

if [[ ! -f "${TRAIN_JSON}" || ! -f "${VAL_JSON}" || ! -f "${TEST_JSON}" ]]; then
    echo "Reviewer finetuning JSON files are missing."
    echo "Please prepare them first with:"
    if [[ -n "${POS_RATIO}" ]]; then
        echo "  ./data_process/rebucket_reviewer_data.sh ${DATASET_NAME} ${RESULT_MODEL_NAME} ${LENGTH} ${POS_RATIO}"
        echo "  ./data_process/rebucket_reviewer_data_oof_train.sh ${DATASET_NAME} ${RESULT_MODEL_NAME} ${LENGTH} ${POS_RATIO}"
    else
        echo "  ./data_process/rebucket_reviewer_data_oof_train.sh ${DATASET_NAME} ${RESULT_MODEL_NAME} ${LENGTH}"
    fi
    echo "Expected:"
    echo "  ${TRAIN_JSON}"
    echo "  ${VAL_JSON}"
    echo "  ${TEST_JSON}"
    exit 1
fi

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${CUDA}" python LLM/finetuning_test.py \
    --use_peft \
    --peft_method lora \
    --batch_size_training $BATCH_SIZE \
    --val_batch_size $BATCH_SIZE \
    --context_length "${CONTEXT_LENGTH}" \
    --num_epochs 5 \
    --model_name ${MODEL_MAP[$MODEL_NAME]} \
    --alpaca_dataset.train_data_path "${TRAIN_JSON}" \
    --alpaca_dataset.valid_data_path "${VAL_JSON}" \
    --output_dir "${OUTPUT_DIR}" \
    >"${OUTPUT_DIR}/finetuning_${MODEL_NAME}_lora_${OUTPUT_STEM}.log"
