#!/bin/bash
DATASET_NAME=$1
RESULT_MODEL_NAME=$2
LENGTH=$3
POS_RATIO=$4
CUDA=${5:-"0"}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

if [ $# -lt 4 ]; then
    echo "Usage: $0 <DATASET_NAME> <RESULT_MODEL_NAME> <LENGTH> <POS_RATIO> [CUDA]"
    exit 1
fi

if [[ ${BASH_VERSINFO[0]} -lt 4 ]]; then
    echo "Bash version 4.0 or higher is required."
    exit 1
fi

REVIEWER_DATA_DIR="reviewer_finetune_data"
DATASET_TAG="${DATASET_NAME}_${LENGTH}_${POS_RATIO}"
REVIEWER_DATA_SUBDIR="${REVIEWER_DATA_DIR}/${RESULT_MODEL_NAME}_imbalance/${DATASET_TAG}"
REBUCKETED_DATA_SUBDIR="${REVIEWER_DATA_DIR}/${RESULT_MODEL_NAME}_imbalance/${DATASET_TAG}_length_rebucketed"

echo "Dataset: ${DATASET_NAME}"
echo "Reviewer result model: ${RESULT_MODEL_NAME}"
echo "Length: ${LENGTH}"
echo "POS_RATIO: ${POS_RATIO}"
echo "CUDA: ${CUDA}"

python3 "${REPO_ROOT}/scripts/generate_reviewer_finetune_json.py" \
    --dataset-name "${DATASET_NAME}" \
    --result-model "${RESULT_MODEL_NAME}" \
    --length "${LENGTH}" \
    --pos-ratio "${POS_RATIO}" \
    --repo-root "${REPO_ROOT}" \
    --output-root "${REVIEWER_DATA_DIR}"

venv/bin/python "${REPO_ROOT}/scripts/rebucket_reviewer_json_by_length.py" \
    --input-dir "${REPO_ROOT}/${REVIEWER_DATA_SUBDIR}"

echo "Rebucketed reviewer data is ready under:"
echo "  ${REBUCKETED_DATA_SUBDIR}"
