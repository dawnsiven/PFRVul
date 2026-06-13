#!/bin/bash
set -euo pipefail

DATASET_NAME=${1:-}
RESULT_MODEL_NAME=${2:-}
MODEL_NAME=${3:-}
LENGTH=${4:-}
POS_RATIO=""
LENGTH_BUCKET="0-512"
REQUESTED_CHECKPOINT=""
CUDA="0"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# Check if the required parameters are provided
if [ $# -lt 4 ]; then
    echo "Usage (imbalance): $0 <DATASET_NAME> <RESULT_MODEL_NAME> <LLM_MODEL_NAME> <LENGTH> <POS_RATIO> [LENGTH_BUCKET] [EPOCH|epoch-N] [CUDA]"
    echo "Usage (non-imbalance): $0 <DATASET_NAME> <RESULT_MODEL_NAME> <LLM_MODEL_NAME> <LENGTH> [LENGTH_BUCKET] [EPOCH|epoch-N] [CUDA]"
    exit 1
fi

# Parsing rules:
# - non-imbalance:
#   $5 = optional LENGTH_BUCKET (usually contains '-')
# - imbalance:
#   $5 = POS_RATIO (numeric)
#   $6 = optional LENGTH_BUCKET
if [ $# -ge 5 ]; then
    if [[ "$5" =~ ^[0-9]+$ ]]; then
        POS_RATIO="$5"
        LENGTH_BUCKET=${6:-"0-512"}
        REQUESTED_CHECKPOINT=${7:-""}
        CUDA=${8:-"0"}
    else
        LENGTH_BUCKET="$5"
        REQUESTED_CHECKPOINT=${6:-""}
        CUDA=${7:-"0"}
    fi
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

if [[ -z "${POS_RATIO}" && $# -eq 6 && "${REQUESTED_CHECKPOINT}" =~ ^[0-9]+$ ]]; then
    CUDA="${REQUESTED_CHECKPOINT}"
    REQUESTED_CHECKPOINT=""
fi

if [[ -n "${POS_RATIO}" && $# -eq 7 && "${REQUESTED_CHECKPOINT}" =~ ^[0-9]+$ ]]; then
    CUDA="${REQUESTED_CHECKPOINT}"
    REQUESTED_CHECKPOINT=""
fi

if [[ -n "${REQUESTED_CHECKPOINT}" && "${REQUESTED_CHECKPOINT}" =~ ^[0-9]+$ ]]; then
    REQUESTED_CHECKPOINT="epoch-${REQUESTED_CHECKPOINT}"
fi

REVIEWER_DATA_DIR="reviewer_finetune_data"
DATASET_TAG="${DATASET_NAME}_${LENGTH}"
REBUCKETED_DATA_SUBDIR="${REVIEWER_DATA_DIR}/${RESULT_MODEL_NAME}/${DATASET_TAG}_length_rebucketed"
OUTPUT_STEM="${RESULT_MODEL_NAME}_${DATASET_TAG}_${LENGTH_BUCKET}"
ORIGINAL_RESULTS_DIR="outputs/${RESULT_MODEL_NAME}/${DATASET_TAG}"

if [[ -n "${POS_RATIO}" ]]; then
    DATASET_TAG="${DATASET_NAME}_${LENGTH}_${POS_RATIO}"
    REBUCKETED_DATA_SUBDIR="${REVIEWER_DATA_DIR}/${RESULT_MODEL_NAME}_imbalance/${DATASET_TAG}_length_rebucketed"
    OUTPUT_STEM="${RESULT_MODEL_NAME}_imbalance_${DATASET_TAG}_${LENGTH_BUCKET}"
    ORIGINAL_RESULTS_DIR="outputs/${RESULT_MODEL_NAME}_imbalance/${DATASET_TAG}"
fi

TEST_JSON="${REBUCKETED_DATA_SUBDIR}/test_${LENGTH_BUCKET}.json"
OUTPUT_DIR="outputs/${MODEL_NAME}_lora/${OUTPUT_STEM}/"
CSV_PATH="${OUTPUT_DIR}/results.csv"
LOG_PATH="${OUTPUT_DIR}/inference_${MODEL_NAME}_lora_${OUTPUT_STEM}.log"
ORIGINAL_RESULTS_CSV="${ORIGINAL_RESULTS_DIR}/results.csv"

find_latest_checkpoint() {
    local output_dir="$1"
    local latest_dir=""
    local latest_epoch=-1
    local checkpoint_dir

    shopt -s nullglob
    for checkpoint_dir in "${output_dir}"/epoch-*; do
        if [[ ! -d "${checkpoint_dir}" ]]; then
            continue
        fi

        local checkpoint_name="${checkpoint_dir##*/}"
        local epoch_suffix="${checkpoint_name#epoch-}"
        if [[ "${epoch_suffix}" =~ ^[0-9]+$ ]] && (( epoch_suffix > latest_epoch )); then
            latest_epoch=${epoch_suffix}
            latest_dir="${checkpoint_dir}"
        fi
    done
    shopt -u nullglob

    if [[ -n "${latest_dir}" ]]; then
        echo "${latest_dir}"
        return
    fi

    if [[ -f "${output_dir}/adapter_config.json" ]]; then
        echo "${output_dir}"
    fi
}

resolve_output_dir() {
    local preferred_dir="$1"
    local search_root
    local candidate_dir
    local matched_dir=""
    local matched_epoch=-1

    if [[ -n "$(find_latest_checkpoint "${preferred_dir}")" ]]; then
        echo "${preferred_dir}"
        return
    fi

    search_root=$(dirname "${preferred_dir}")
    if [[ ! -d "${search_root}" ]]; then
        return
    fi

    shopt -s nullglob
    local patterns=()
    if [[ -n "${POS_RATIO}" ]]; then
        patterns+=("${search_root}/${RESULT_MODEL_NAME}_imbalance_*_${LENGTH_BUCKET}")
    else
        patterns+=("${search_root}/${RESULT_MODEL_NAME}_*_${LENGTH_BUCKET}")
    fi

    shopt -s nullglob
    for candidate_pattern in "${patterns[@]}"; do
        for candidate_dir in ${candidate_pattern}; do
        if [[ ! -d "${candidate_dir}" ]]; then
            continue
        fi

        local checkpoint_dir
        checkpoint_dir=$(find_latest_checkpoint "${candidate_dir}")
        if [[ -z "${checkpoint_dir}" ]]; then
            continue
        fi

        local checkpoint_name="${checkpoint_dir##*/}"
        local epoch_suffix=-1
        if [[ "${checkpoint_name}" =~ ^epoch-([0-9]+)$ ]]; then
            epoch_suffix="${BASH_REMATCH[1]}"
        elif [[ -f "${checkpoint_dir}/adapter_config.json" ]]; then
            epoch_suffix=999999
        fi

        if (( epoch_suffix > matched_epoch )); then
            matched_epoch=${epoch_suffix}
            matched_dir="${candidate_dir}"
        fi
        done
    done
    shopt -u nullglob

    if [[ -n "${matched_dir}" ]]; then
        echo "${matched_dir}"
    fi
}

resolve_requested_checkpoint() {
    local output_dir="$1"
    local requested_checkpoint="$2"
    local checkpoint_dir="${output_dir}/${requested_checkpoint}"

    if [[ -d "${checkpoint_dir}" ]]; then
        echo "${checkpoint_dir}"
    fi
}

if [[ ! -f "${TEST_JSON}" ]]; then
    echo "Rebucketed reviewer test JSON is missing: ${TEST_JSON}"
    echo "Please prepare it first with:"
    if [[ -n "${POS_RATIO}" ]]; then
        echo "  ./data_process/rebucket_reviewer_data.sh ${DATASET_NAME} ${RESULT_MODEL_NAME} ${LENGTH} ${POS_RATIO}"
        echo "  ./data_process/rebucket_reviewer_data_oof_train.sh ${DATASET_NAME} ${RESULT_MODEL_NAME} ${LENGTH} ${POS_RATIO}"
    else
        echo "  ./data_process/rebucket_reviewer_data_oof_train.sh ${DATASET_NAME} ${RESULT_MODEL_NAME} ${LENGTH}"
    fi
    exit 1
fi

RESOLVED_OUTPUT_DIR=$(resolve_output_dir "${OUTPUT_DIR}")
if [[ -z "${RESOLVED_OUTPUT_DIR}" ]]; then
    echo "No LoRA output directory with checkpoints found."
    echo "Preferred directory: ${OUTPUT_DIR}"
    echo "Searched under: outputs/${MODEL_NAME}_lora/"
    echo "Please run scripts/finetune_test.sh first, or check whether the actual output directory name differs."
    exit 1
fi

if [[ "${RESOLVED_OUTPUT_DIR}" != "${OUTPUT_DIR}" ]]; then
    echo "Preferred output directory has no checkpoints."
    echo "Falling back to detected checkpoint directory: ${RESOLVED_OUTPUT_DIR}"
fi

OUTPUT_DIR="${RESOLVED_OUTPUT_DIR}"
CSV_PATH="${OUTPUT_DIR}/results.csv"
LOG_PATH="${OUTPUT_DIR}/inference_${MODEL_NAME}_lora_${OUTPUT_STEM}.log"

CHECKPOINT_DIR=$(find_latest_checkpoint "${OUTPUT_DIR}")
if [[ -n "${REQUESTED_CHECKPOINT}" ]]; then
    CHECKPOINT_DIR=$(resolve_requested_checkpoint "${OUTPUT_DIR}" "${REQUESTED_CHECKPOINT}")
    if [[ -z "${CHECKPOINT_DIR}" ]]; then
        echo "Requested checkpoint was not found: ${OUTPUT_DIR}/${REQUESTED_CHECKPOINT}"
        exit 1
    fi
fi

if [[ -z "${CHECKPOINT_DIR}" ]]; then
    echo "No LoRA checkpoint found under: ${OUTPUT_DIR}"
    echo "Expected an epoch directory such as ${OUTPUT_DIR}/epoch-4"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Reviewer result model: ${RESULT_MODEL_NAME}"
echo "Length: ${LENGTH}"
if [[ -n "${POS_RATIO}" ]]; then
    echo "POS_RATIO: ${POS_RATIO}"
else
    echo "POS_RATIO: <non-imbalance mode>"
fi
echo "Length bucket: ${LENGTH_BUCKET}"
if [[ -n "${REQUESTED_CHECKPOINT}" ]]; then
    echo "Requested checkpoint: ${REQUESTED_CHECKPOINT}"
fi
echo "Using checkpoint: ${CHECKPOINT_DIR}"
echo "Test JSON: ${TEST_JSON}"

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${CUDA}" python LLM/inference.py \
    --base_model "${MODEL_MAP[$MODEL_NAME]}" \
    --tuned_model "${CHECKPOINT_DIR}" \
    --data_file "${TEST_JSON}" \
    --csv_path "${CSV_PATH}" \
    >"${LOG_PATH}"

if [[ -f "${ORIGINAL_RESULTS_CSV}" ]]; then
    echo "Merging reviewer results back into original small-model results..."
    python3 scripts/merge_reviewer_lora_results.py \
        --original_results_csv "${ORIGINAL_RESULTS_CSV}" \
        --reviewer_results_csv "${CSV_PATH}" \
        --output_dir "${OUTPUT_DIR}"
else
    echo "Original small-model results.csv not found, skipping merge:"
    echo "  ${ORIGINAL_RESULTS_CSV}"
fi
