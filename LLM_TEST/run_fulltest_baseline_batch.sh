#!/usr/bin/env bash

set -euo pipefail

PROMPT_FILE="${PROMPT_FILE:-LLM_TEST/Prompt/r-b.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-LLM_TEST/output}"
WORKERS="${WORKERS:-10}"
TEMPERATURE="${TEMPERATURE:-0.5}"
LIMIT="${LIMIT:-}"
RESPONSE_FORMAT="${RESPONSE_FORMAT:-json_object}"

DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
MOONSHOT_API_BASE="${MOONSHOT_API_BASE:-https://api.moonshot.cn/v1}"
MOONSHOT_MODEL="${MOONSHOT_MODEL:-moonshot-v1-8k}"

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "DEEPSEEK_API_KEY is not set."
    exit 1
fi

if [[ -z "${MOONSHOT_API_KEY:-}" ]]; then
    echo "MOONSHOT_API_KEY is not set."
    exit 1
fi

if [[ ! -f "${PROMPT_FILE}" ]]; then
    echo "Prompt file not found: ${PROMPT_FILE}"
    exit 1
fi

DATASETS=(
    "bigvul_0-512_prepared"
    "diversevul_0-512_prepared"
    "devign_0-512_prepared"
    "fulltest_chunk_lable_0_512_chunks_llm_only"
)

build_command() {
    local input_json="$1"
    local api_base="$2"
    local api_key="$3"
    local model="$4"
    local output_name="$5"

    local cmd=(
        python3 LLM_TEST/llm_api_judge.py
        --input_json "${input_json}"
        --prompt_file "${PROMPT_FILE}"
        --api_base "${api_base}"
        --api_key "${api_key}"
        --model "${model}"
        --output_root "${OUTPUT_ROOT}"
        --output_name "${output_name}"
        --workers "${WORKERS}"
        --temperature "${TEMPERATURE}"
        --response_format "${RESPONSE_FORMAT}"
        --resume
    )

    if [[ -n "${LIMIT}" ]]; then
        cmd+=(--limit "${LIMIT}")
    fi

    printf '%q ' "${cmd[@]}"
    printf '\n'
}

build_eval_command() {
    local llm_predictions_csv="$1"
    local output_dir="$2"

    local cmd=(
        python3 LLM_TEST/evaluate_llm_full_dataset.py
        --llm_predictions_csv "${llm_predictions_csv}"
        --output_dir "${output_dir}"
    )

    printf '%q ' "${cmd[@]}"
    printf '\n'
}

run_model() {
    local provider="$1"
    local api_base="$2"
    local api_key="$3"
    local model="$4"

    for dataset in "${DATASETS[@]}"; do
        local input_json="LLM_TEST/intermediate/${dataset}/full_test_samples.jsonl"
        local output_name="${dataset}/${model}_$(basename "${PROMPT_FILE}" .txt)"
        local output_dir="${OUTPUT_ROOT}/${output_name}"
        local llm_predictions_csv="${output_dir}/llm_predictions.csv"
        local cmd=(
            python3 LLM_TEST/llm_api_judge.py
            --input_json "${input_json}"
            --prompt_file "${PROMPT_FILE}"
            --api_base "${api_base}"
            --api_key "${api_key}"
            --model "${model}"
            --output_root "${OUTPUT_ROOT}"
            --output_name "${output_name}"
            --workers "${WORKERS}"
            --temperature "${TEMPERATURE}"
            --response_format "${RESPONSE_FORMAT}"
            --resume
        )

        if [[ ! -f "${input_json}" ]]; then
            echo "Input JSON not found, skip: ${input_json}"
            continue
        fi

        if [[ -n "${LIMIT}" ]]; then
            cmd+=(--limit "${LIMIT}")
        fi

        echo "================================================================"
        echo "provider=${provider}"
        echo "dataset=${dataset}"
        echo "model=${model}"
        echo "prompt_file=${PROMPT_FILE}"
        echo "output_name=${output_name}"
        echo "================================================================"

        build_command "${input_json}" "${api_base}" "${api_key}" "${model}" "${output_name}"
        "${cmd[@]}"

        if [[ ! -f "${llm_predictions_csv}" ]]; then
            echo "llm_predictions.csv not found after judging: ${llm_predictions_csv}"
            continue
        fi

        echo "evaluate_output_dir=${output_dir}"
        build_eval_command "${llm_predictions_csv}" "${output_dir}"
        python3 LLM_TEST/evaluate_llm_full_dataset.py \
            --llm_predictions_csv "${llm_predictions_csv}" \
            --output_dir "${output_dir}"
    done
}

run_model "deepseek" "${DEEPSEEK_API_BASE}" "${DEEPSEEK_API_KEY}" "${DEEPSEEK_MODEL}"
run_model "moonshot" "${MOONSHOT_API_BASE}" "${MOONSHOT_API_KEY}" "${MOONSHOT_MODEL}"
