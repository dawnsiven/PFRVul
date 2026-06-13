#!/bin/bash
DATASET_NAME=$1
MODEL_NAME=$2
LENGTH=$3
POS_RATIO=$4
CUDA=${5:-"0"}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
LOCAL_MODEL_ROOT="${REPO_ROOT}/model"

# Check if the first three parameters are provided
if [ $# -lt 4 ]; then
    echo "Usage: $0 <DATASET_NAME> <MODEL_NAME> <LENGTH> <POS_RATIO> [CUDA]"
    exit 1
fi

if [[ ${BASH_VERSINFO[0]} -lt 4 ]]; then
    echo "Bash version 4.0 or higher is required."
    exit 1
fi

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

mkdir -p "outputs/${MODEL_NAME}_imbalance_test/${DATASET_NAME}_${LENGTH}_${POS_RATIO}/"

echo "POS_RATIO: $(echo $POS_RATIO)"

OUTPUT_DIR="outputs/${MODEL_NAME}_imbalance_test/${DATASET_NAME}_${LENGTH}_${POS_RATIO}"
ORIGINAL_TRAIN_DATA_FILE="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_train.json"
TRAIN_DATA_FILE="${ORIGINAL_TRAIN_DATA_FILE}"
EVAL_DATA_FILE="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_validate.json"
TEST_DATA_FILE="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_test.json"
LEGACY_RESULTS_CSV="${OUTPUT_DIR}/results.csv"

echo "Using original train file: ${TRAIN_DATA_FILE}"
echo "Weighted train workflow is disabled in this script."

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
    local log_path="${OUTPUT_DIR}/reviewer_${split_name}_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}_${POS_RATIO}.log"

    echo "Generating reviewer_${split_name}.csv from ${split_data_file}"

    if [[ "$MODEL_NAME" == "GraphCodeBERT" ]]; then
        CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/run.py \
            --output_dir="${OUTPUT_DIR}/" \
            --csv_path="${csv_path}" \
            --tokenizer_name="${TOKENIZER_SOURCE}" \
            --model_name_or_path="${BACKBONE_SOURCE}" \
            --do_test \
            --train_data_file="${TRAIN_DATA_FILE}" \
            --eval_data_file="${EVAL_DATA_FILE}" \
            --test_data_file="${split_data_file}" \
            --block_size 512 \
            --eval_batch_size 4 \
            --seed 42 \
            2>"${log_path}"
    elif [[ "$MODEL_NAME" == "UniXcoder" ]]; then
        CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/run.py \
            --output_dir="${OUTPUT_DIR}/" \
            --csv_path="${csv_path}" \
            --model_name_or_path="${BACKBONE_SOURCE}" \
            --do_test \
            --train_data_file="${TRAIN_DATA_FILE}" \
            --eval_data_file="${EVAL_DATA_FILE}" \
            --test_data_file="${split_data_file}" \
            --block_size 512 \
            --eval_batch_size 4 \
            --seed 42 \
            2>"${log_path}"
    else
        CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/run.py \
            --output_dir="${OUTPUT_DIR}/" \
            --csv_path="${csv_path}" \
            --model_type=roberta \
            --tokenizer_name="${TOKENIZER_SOURCE}" \
            --model_name_or_path="${BACKBONE_SOURCE}" \
            --do_test \
            --train_data_file="${TRAIN_DATA_FILE}" \
            --eval_data_file="${EVAL_DATA_FILE}" \
            --test_data_file="${split_data_file}" \
            --block_size 512 \
            --eval_batch_size 4 \
            --seed 42 \
            2>"${log_path}"
    fi
}

if [[ "$MODEL_NAME" == "GraphCodeBERT" ]]; then
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/run.py \
    --output_dir="${OUTPUT_DIR}/" \
    --csv_path="${OUTPUT_DIR}/reviewer_test.csv" \
    --tokenizer_name="${TOKENIZER_SOURCE}" \
    --model_name_or_path="${BACKBONE_SOURCE}" \
    --do_train \
    --do_eval \
    --train_data_file="${TRAIN_DATA_FILE}" \
    --eval_data_file="${EVAL_DATA_FILE}" \
    --test_data_file="${TEST_DATA_FILE}" \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size  4\ 
    --eval_batch_size 4 \
    --evaluate_during_training \
    --seed 42 \
    2>"${OUTPUT_DIR}/train_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}_${POS_RATIO}.log"
elif [[ "$MODEL_NAME" == "UniXcoder" ]]; then
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/run.py \
    --output_dir="${OUTPUT_DIR}/" \
    --csv_path="${OUTPUT_DIR}/reviewer_test.csv" \
    --model_name_or_path="${BACKBONE_SOURCE}" \
    --do_train \
    --do_eval \
    --train_data_file="${TRAIN_DATA_FILE}" \
    --eval_data_file="${EVAL_DATA_FILE}" \
    --test_data_file="${TEST_DATA_FILE}" \
    --num_train_epochs 5 \
    --block_size 512 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --seed 42 \
    2>"${OUTPUT_DIR}/train_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}_${POS_RATIO}.log"
else
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/run.py \
    --output_dir="${OUTPUT_DIR}/" \
    --csv_path="${OUTPUT_DIR}/reviewer_test.csv" \
    --model_type=roberta \
    --tokenizer_name="${TOKENIZER_SOURCE}" \
    --model_name_or_path="${BACKBONE_SOURCE}" \
    --do_train \
    --do_eval \
    --train_data_file="${TRAIN_DATA_FILE}" \
    --eval_data_file="${EVAL_DATA_FILE}" \
    --test_data_file="${TEST_DATA_FILE}" \
    --epoch 5 \
    --block_size 512 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --evaluate_during_training \
    --seed 42 \
    2>"${OUTPUT_DIR}/train_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}_${POS_RATIO}.log"
fi

run_reviewer_inference "train" "${ORIGINAL_TRAIN_DATA_FILE}"
run_reviewer_inference "val" "${EVAL_DATA_FILE}"
run_reviewer_inference "test" "${TEST_DATA_FILE}"

cp "${OUTPUT_DIR}/reviewer_test.csv" "${LEGACY_RESULTS_CSV}"
echo "Copied reviewer_test.csv to legacy results.csv for compatibility."
