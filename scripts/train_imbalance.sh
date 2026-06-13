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

mkdir -p "outputs/${MODEL_NAME}_imbalance/${DATASET_NAME}_${LENGTH}_${POS_RATIO}/"

echo "POS_RATIO: $(echo $POS_RATIO)"

BACKBONE_SOURCE=$(resolve_model_source "microsoft/codebert-base")
TOKENIZER_SOURCE="$BACKBONE_SOURCE"

if [[ "$BACKBONE_SOURCE" == "microsoft/codebert-base" ]]; then
    echo "Using remote backbone: $BACKBONE_SOURCE"
else
    echo "Using local backbone: $BACKBONE_SOURCE"
fi

if [[ "$MODEL_NAME" == "GraphCodeBERT" ]]; then
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/run.py \
    --output_dir="outputs/${MODEL_NAME}_imbalance/${DATASET_NAME}_${LENGTH}_${POS_RATIO}/" \
    --csv_path="outputs/${MODEL_NAME}_imbalance/${DATASET_NAME}_${LENGTH}_${POS_RATIO}/results.csv" \
    --tokenizer_name="${TOKENIZER_SOURCE}" \
    --model_name_or_path="${BACKBONE_SOURCE}" \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_train.json" \
    --eval_data_file="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_validate.json" \
    --test_data_file="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_test.json" \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size  4\ 
    --eval_batch_size 4 \
    --evaluate_during_training \
    --seed 42 \
    2>"outputs/${MODEL_NAME}_imbalance/${DATASET_NAME}_${LENGTH}_${POS_RATIO}/train_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}_${POS_RATIO}.log"
elif [[ "$MODEL_NAME" == "UniXcoder" ]]; then
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/run.py \
    --output_dir="outputs/${MODEL_NAME}_imbalance/${DATASET_NAME}_${LENGTH}_${POS_RATIO}/" \
    --csv_path="outputs/${MODEL_NAME}_imbalance/${DATASET_NAME}_${LENGTH}_${POS_RATIO}/results.csv" \
    --model_name_or_path="${BACKBONE_SOURCE}" \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_train.json" \
    --eval_data_file="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_validate.json" \
    --test_data_file="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_test.json" \
    --num_train_epochs 10 \
    --block_size 512 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --seed 42 \
    2>"outputs/${MODEL_NAME}_imbalance/${DATASET_NAME}_${LENGTH}_${POS_RATIO}/train_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}_${POS_RATIO}.log"
else
CUDA_VISIBLE_DEVICES="${CUDA}" python ${MODEL_NAME}/run.py \
    --output_dir="outputs/${MODEL_NAME}_imbalance/${DATASET_NAME}_${LENGTH}_${POS_RATIO}/" \
    --csv_path="outputs/${MODEL_NAME}_imbalance/${DATASET_NAME}_${LENGTH}_${POS_RATIO}/results.csv" \
    --model_type=roberta \
    --tokenizer_name="${TOKENIZER_SOURCE}" \
    --model_name_or_path="${BACKBONE_SOURCE}" \
    --do_train \
    --do_eval \
    --do_test \
    --train_data_file="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_train.json" \
    --eval_data_file="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_validate.json" \
    --test_data_file="data/${DATASET_NAME}_subsampled/alpaca/${DATASET_NAME}_${LENGTH}_${POS_RATIO}_test.json" \
    --epoch 10 \
    --block_size 512 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --evaluate_during_training \
    --seed 42 \
    2>"outputs/${MODEL_NAME}_imbalance/${DATASET_NAME}_${LENGTH}_${POS_RATIO}/train_${MODEL_NAME}_${DATASET_NAME}_${LENGTH}_${POS_RATIO}.log"
fi
