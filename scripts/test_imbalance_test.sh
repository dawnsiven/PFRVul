#!/bin/bash
DATASET_NAME=$1
MODEL_NAME=$2
LENGTH=$3
POS_RATIO=""
CUDA="0"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
LOCAL_MODEL_ROOT="${REPO_ROOT}/model"

# Support both:
# 1) imbalance datasets: <DATASET_NAME> <MODEL_NAME> <LENGTH> <POS_RATIO> [CUDA]
# 2) standard datasets:   <DATASET_NAME> <MODEL_NAME> <LENGTH> [CUDA]
if [ $# -lt 3 ]; then
    echo "Usage: $0 <DATASET_NAME> <MODEL_NAME> <LENGTH> [POS_RATIO] [CUDA]"
    exit 1
fi

if [ $# -eq 4 ]; then
    if [[ "$4" =~ ^[0-9]+$ ]]; then
        CUDA="$4"
    else
        POS_RATIO="$4"
    fi
elif [ $# -ge 5 ]; then
    POS_RATIO="$4"
    CUDA="${5:-0}"
fi

if [[ ${BASH_VERSINFO[0]} -lt 4 ]]; then
    echo "Bash version 4.0 or higher is required."
    exit 1
fi

if [[ -x "${REPO_ROOT}/venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/venv/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "Neither python nor python3 is available in PATH."
    exit 1
fi

echo "Using Python interpreter: ${PYTHON_BIN}"

resolve_model_source() {
    local model_name="$1"
    local direct_path="${model_name/#\~/$HOME}"
    if [[ -e "$direct_path" ]]; then
        echo "$direct_path"
        return
    fi

    if [[ ! -d "$LOCAL_MODEL_ROOT" ]]; then
        echo "$model_name"
        return
    fi

    local normalized="${model_name//\\//}"
    normalized="${normalized#/}"
    local base_name="${normalized##*/}"
    local candidates=(
        "$LOCAL_MODEL_ROOT/$base_name"
        "$LOCAL_MODEL_ROOT/$normalized"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -e "$candidate" ]]; then
            echo "$candidate"
            return
        fi
    done

    echo "$model_name"
}

BLOCK_SIZE=$(echo "$LENGTH" | awk -F'-' '{print $2}')
if [[ -z "$BLOCK_SIZE" ]]; then
    BLOCK_SIZE=512
fi

DATASET_TAG="${DATASET_NAME}_${LENGTH}"
DATASET_DIR="data/${DATASET_NAME}/alpaca"
TRAIN_DATA_FILE="${DATASET_DIR}/${DATASET_NAME}_${LENGTH}_train.json"
EVAL_DATA_FILE="${DATASET_DIR}/${DATASET_NAME}_${LENGTH}_validate.json"
TEST_DATA_FILE="${DATASET_DIR}/${DATASET_NAME}_${LENGTH}_test.json"
PRIMARY_OUTPUT_DIR="outputs/${MODEL_NAME}/${DATASET_TAG}"
FALLBACK_OUTPUT_DIR=""

if [[ -n "${POS_RATIO}" ]]; then
    DATASET_TAG="${DATASET_NAME}_${LENGTH}_${POS_RATIO}"
    DATASET_DIR="data/${DATASET_NAME}_subsampled/alpaca"
    TRAIN_DATA_FILE="${DATASET_DIR}/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_train.json"
    EVAL_DATA_FILE="${DATASET_DIR}/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_validate.json"
    TEST_DATA_FILE="${DATASET_DIR}/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_test.json"
    PRIMARY_OUTPUT_DIR="outputs/${MODEL_NAME}_imbalance/${DATASET_TAG}"
    FALLBACK_OUTPUT_DIR="outputs/${MODEL_NAME}_imbalance_test/${DATASET_TAG}"
fi

if [[ ! -f "${TRAIN_DATA_FILE}" || ! -f "${EVAL_DATA_FILE}" || ! -f "${TEST_DATA_FILE}" ]]; then
    echo "Dataset files not found for ${DATASET_TAG}."
    echo "Expected:"
    echo "  ${TRAIN_DATA_FILE}"
    echo "  ${EVAL_DATA_FILE}"
    echo "  ${TEST_DATA_FILE}"
    exit 1
fi

if [[ -d "${PRIMARY_OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${PRIMARY_OUTPUT_DIR}"
elif [[ -n "${FALLBACK_OUTPUT_DIR}" && -d "${FALLBACK_OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${FALLBACK_OUTPUT_DIR}"
else
    echo "No existing output directory found for ${DATASET_TAG}."
    echo "Expected:"
    echo "  ${PRIMARY_OUTPUT_DIR}"
    if [[ -n "${FALLBACK_OUTPUT_DIR}" ]]; then
        echo "  ${FALLBACK_OUTPUT_DIR}"
    fi
    exit 1
fi

echo "Dataset tag: ${DATASET_TAG}"
if [[ -n "${POS_RATIO}" ]]; then
    echo "POS_RATIO: ${POS_RATIO}"
fi
echo "Using existing output directory: ${OUTPUT_DIR}"

LEGACY_RESULTS_CSV="${OUTPUT_DIR}/results.csv"
BACKBONE_SOURCE=$(resolve_model_source "microsoft/codebert-base")
TOKENIZER_SOURCE="$BACKBONE_SOURCE"

if [[ "$BACKBONE_SOURCE" == "microsoft/codebert-base" ]]; then
    echo "Using remote backbone: $BACKBONE_SOURCE"
else
    echo "Using local backbone: $BACKBONE_SOURCE"
fi

run_reviewer_inference() {
    local split_name="$1"
    local split_data_file="$2"
    local csv_path="${OUTPUT_DIR}/reviewer_${split_name}.csv"
    local log_path="${OUTPUT_DIR}/reviewer_${split_name}_${MODEL_NAME}_${DATASET_TAG}.log"

    echo "Generating reviewer_${split_name}.csv from ${split_data_file}"

    if [[ "$MODEL_NAME" == "GraphCodeBERT" ]]; then
        CUDA_VISIBLE_DEVICES="${CUDA}" "${PYTHON_BIN}" ${MODEL_NAME}/run.py \
            --output_dir="${OUTPUT_DIR}/" \
            --csv_path="${csv_path}" \
            --tokenizer_name="${TOKENIZER_SOURCE}" \
            --model_name_or_path="${BACKBONE_SOURCE}" \
            --do_test \
            --train_data_file="${TRAIN_DATA_FILE}" \
            --eval_data_file="${EVAL_DATA_FILE}" \
            --test_data_file="${split_data_file}" \
            --block_size "${BLOCK_SIZE}" \
            --eval_batch_size 6 \
            --seed 42 \
            2>"${log_path}"
    elif [[ "$MODEL_NAME" == "UniXcoder" ]]; then
        CUDA_VISIBLE_DEVICES="${CUDA}" "${PYTHON_BIN}" ${MODEL_NAME}/run.py \
            --output_dir="${OUTPUT_DIR}/" \
            --csv_path="${csv_path}" \
            --model_name_or_path="${BACKBONE_SOURCE}" \
            --do_test \
            --train_data_file="${TRAIN_DATA_FILE}" \
            --eval_data_file="${EVAL_DATA_FILE}" \
            --test_data_file="${split_data_file}" \
            --block_size "${BLOCK_SIZE}" \
            --eval_batch_size 6 \
            --seed 42 \
            2>"${log_path}"
    else
        CUDA_VISIBLE_DEVICES="${CUDA}" "${PYTHON_BIN}" ${MODEL_NAME}/run.py \
            --output_dir="${OUTPUT_DIR}/" \
            --csv_path="${csv_path}" \
            --model_type=roberta \
            --tokenizer_name="${TOKENIZER_SOURCE}" \
            --model_name_or_path="${BACKBONE_SOURCE}" \
            --do_test \
            --train_data_file="${TRAIN_DATA_FILE}" \
            --eval_data_file="${EVAL_DATA_FILE}" \
            --test_data_file="${split_data_file}" \
            --block_size "${BLOCK_SIZE}" \
            --eval_batch_size 6 \
            --seed 42 \
            2>"${log_path}"
    fi
}

run_reviewer_inference "train" "${TRAIN_DATA_FILE}"
run_reviewer_inference "val" "${EVAL_DATA_FILE}"
run_reviewer_inference "test" "${TEST_DATA_FILE}"

if [[ ! -f "${OUTPUT_DIR}/reviewer_test.csv" ]]; then
    echo "reviewer_test.csv was not generated. Check reviewer logs under ${OUTPUT_DIR}."
    exit 1
fi

cp "${OUTPUT_DIR}/reviewer_test.csv" "${LEGACY_RESULTS_CSV}"
echo "Copied reviewer_test.csv to legacy results.csv for compatibility."
