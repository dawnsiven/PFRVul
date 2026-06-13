#!/bin/bash
set -euo pipefail

print_usage() {
    cat <<'EOF'
Usage:
  Legacy mode:
    bash LLM_TEST/run_review.sh [config_path] [env_path] [limit]

  Auto mode:
    bash LLM_TEST/run_review.sh \
      --result-model llama3.2_lora_imbalance \
      --dataset bigvul_1 \
      --prob-range 0.5-0.9 \
      [--limit 100] [--config LLM_TEST/exp.yaml] [--env LLM_TEST/.env]

Auto mode derives:
  - RESULTS_CSV
  - INPUT_JSON
  - output directory names
  - metrics output paths

It still reads review-model settings from the env file, such as:
  - LLM_MODEL
  - API_BASE
  - API_KEY_ENV / OPENAI_API_KEY
  - PROMPT_FILE
  - OUTPUT_ROOT
EOF
}

run_legacy_mode() {
    local config_path="${1:-LLM_TEST/exp.yaml}"
    local env_path="${2:-LLM_TEST/.env}"
    local limit="${3:-100}"

    if [[ ! -f "${config_path}" ]]; then
        echo "Config file not found: ${config_path}"
        exit 1
    fi

    if [[ ! -f "${env_path}" ]]; then
        echo "Env file not found: ${env_path}"
        exit 1
    fi

    set -a
    source "${env_path}"
    set +a

    local input_json_path="${INPUT_JSON:-}"
    local intermediate_root_path="${INTERMEDIATE_ROOT:-LLM_TEST/intermediate}"
    local dataset_id_value="${DATASET_ID:-}"

    if [[ -z "${input_json_path}" ]]; then
        echo "Missing INPUT_JSON in ${env_path}"
        exit 1
    fi

    if [[ ! -f "${input_json_path}" ]]; then
        echo "Input JSON not found: ${input_json_path}"
        echo "Running extract_positive_samples.py to generate it..."
        python3 LLM_TEST/extract_positive_samples.py \
            --config "${config_path}" \
            --env_file "${env_path}"
    fi

    if [[ ! -f "${input_json_path}" && -n "${dataset_id_value}" ]]; then
        local fallback_input_json="${intermediate_root_path%/}/${dataset_id_value}/positive_samples.jsonl"
        if [[ -f "${fallback_input_json}" ]]; then
            echo "Configured INPUT_JSON was not generated."
            echo "Using extracted fallback: ${fallback_input_json}"
            input_json_path="${fallback_input_json}"
        fi
    fi

    if [[ ! -f "${input_json_path}" ]]; then
        echo "Failed to locate input json."
        echo "configured_input_json=${INPUT_JSON:-}"
        if [[ -n "${dataset_id_value}" ]]; then
            echo "fallback_input_json=${intermediate_root_path%/}/${dataset_id_value}/positive_samples.jsonl"
        fi
        echo "Please check RESULTS_CSV / DATA_JSON / INPUT_JSON in ${env_path}"
        exit 1
    fi

    echo "Starting LLM review"
    echo "mode=legacy"
    echo "config=${config_path}"
    echo "env=${env_path}"
    echo "limit=${limit}"
    echo "input_json=${input_json_path}"

    python3 LLM_TEST/llm_api_judge.py \
        --config "${config_path}" \
        --env_file "${env_path}" \
        --input_json "${input_json_path}" \
        --limit "${limit}" \
        --output_by_prompt_version

    python3 LLM_TEST/recompute_metrics.py \
        --config "${config_path}" \
        --env_file "${env_path}"

    echo "Completed."
}

normalize_prob_range() {
    local raw_range="$1"
    local normalized="${raw_range// /}"
    normalized="${normalized//:/-}"
    normalized="${normalized//_/-}"
    echo "${normalized}"
}

derive_output_prompt_tag() {
    local dataset_id="$1"
    local prompt_file_path="$2"
    local prompt_stem
    local version_suffix=""

    prompt_stem="$(basename "${prompt_file_path}")"
    prompt_stem="${prompt_stem%.*}"

    if [[ "${prompt_stem}" =~ ^CWE-[0-9]+(.*)$ ]]; then
        version_suffix="${BASH_REMATCH[1]}"
    fi

    if [[ "${dataset_id}" =~ cwe([0-9]+) ]]; then
        echo "CWE-${BASH_REMATCH[1]}${version_suffix}"
        return 0
    fi

    echo "${prompt_stem}"
}

run_auto_mode() {
    local config_path="LLM_TEST/exp.yaml"
    local env_path="LLM_TEST/.env"
    local limit=""
    local result_model=""
    local dataset_id=""
    local prob_range=""
    local workers="1"
    local data_json_override=""
    local prompt_file_override=""
    local output_root_override=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --result-model)
                result_model="$2"
                shift 2
                ;;
            --dataset)
                dataset_id="$2"
                shift 2
                ;;
            --prob-range)
                prob_range="$2"
                shift 2
                ;;
            --limit)
                limit="$2"
                shift 2
                ;;
            --workers)
                workers="$2"
                shift 2
                ;;
            --config)
                config_path="$2"
                shift 2
                ;;
            --env)
                env_path="$2"
                shift 2
                ;;
            --data-json)
                data_json_override="$2"
                shift 2
                ;;
            --prompt-file)
                prompt_file_override="$2"
                shift 2
                ;;
            --output-root)
                output_root_override="$2"
                shift 2
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown argument: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    if [[ -z "${result_model}" || -z "${dataset_id}" || -z "${prob_range}" ]]; then
        echo "Auto mode requires --result-model, --dataset and --prob-range."
        print_usage
        exit 1
    fi

    if [[ ! -f "${config_path}" ]]; then
        echo "Config file not found: ${config_path}"
        exit 1
    fi

    if [[ ! -f "${env_path}" ]]; then
        echo "Env file not found: ${env_path}"
        exit 1
    fi

    set -a
    source "${env_path}"
    set +a

    local normalized_range
    normalized_range="$(normalize_prob_range "${prob_range}")"
    if [[ ! "${normalized_range}" =~ ^([0-9]+(\.[0-9]+)?)-([0-9]+(\.[0-9]+)?)$ ]]; then
        echo "Invalid --prob-range: ${prob_range}"
        echo "Expected forms like: 0.5-0.9"
        exit 1
    fi

    local min_prob="${BASH_REMATCH[1]}"
    local max_prob="${BASH_REMATCH[3]}"

    local data_root_path="${DATA_ROOT:-data}"
    local intermediate_root_path="${INTERMEDIATE_ROOT:-LLM_TEST/intermediate}"
    local output_root_path="${output_root_override:-${OUTPUT_ROOT:-LLM_TEST/output}}"
    local prompt_file_path="${prompt_file_override:-${PROMPT_FILE:-}}"
    local review_model="${LLM_MODEL:-}"
    local results_csv="outputs/${result_model}/${dataset_id}/results.csv"
    local data_json_path="${data_json_override}"
    local range_tag="${min_prob}_${max_prob}"
    local intermediate_tag="${dataset_id}_${result_model}_positive_${range_tag}"
    local input_json_path="${intermediate_root_path%/}/${intermediate_tag}/positive_samples.jsonl"
    local output_prompt_tag
    local output_tag
    local llm_predictions_csv
    local metrics_output_dir
    local last_run_env_path="LLM_TEST/.env.last_run"

    if [[ -z "${review_model}" ]]; then
        echo "Missing LLM_MODEL in ${env_path}"
        exit 1
    fi

    if [[ ! -f "${results_csv}" ]]; then
        echo "results.csv not found: ${results_csv}"
        exit 1
    fi

    if [[ -z "${prompt_file_path}" ]]; then
        echo "Missing PROMPT_FILE in ${env_path}"
        exit 1
    fi

    if [[ ! -f "${prompt_file_path}" ]]; then
        echo "prompt file not found: ${prompt_file_path}"
        exit 1
    fi

    if [[ -n "${data_json_path}" && ! -f "${data_json_path}" ]]; then
        echo "data json not found: ${data_json_path}"
        exit 1
    fi

    output_prompt_tag="$(derive_output_prompt_tag "${dataset_id}" "${prompt_file_path}")"
    output_tag="${dataset_id}_${result_model}_${review_model}_${output_prompt_tag}"
    llm_predictions_csv="${output_root_path%/}/${output_tag}/llm_predictions.csv"
    metrics_output_dir="${output_root_path%/}/${output_tag}"

    cat > "${last_run_env_path}" <<EOF
RESULTS_CSV=${results_csv}
DATA_JSON=${data_json_path}
INPUT_JSON=${input_json_path}
LLM_PREDICTIONS_CSV=${llm_predictions_csv}
METRICS_OUTPUT_DIR=${metrics_output_dir}
OUTPUT_ROOT=${output_root_path}
PROMPT_FILE=${prompt_file_path}
LLM_MODEL=${review_model}
EOF

    echo "Starting LLM review"
    echo "mode=auto"
    echo "config=${config_path}"
    echo "env=${env_path}"
    echo "limit=${limit:-all}"
    echo "result_model=${result_model}"
    echo "dataset=${dataset_id}"
    echo "prob_range=${min_prob}-${max_prob}"
    echo "workers=${workers}"
    echo "results_csv=${results_csv}"
    echo "input_json=${input_json_path}"
    echo "llm_output_dir=${metrics_output_dir}"
    echo "resolved_env=${last_run_env_path}"

    echo "Refreshing extracted positive samples..."
    python3 LLM_TEST/extract_positive_samples.py \
        --config "${config_path}" \
        --env_file "${env_path}" \
        --results_csv "${results_csv}" \
        --data_json "${data_json_path}" \
        --data_root "${data_root_path}" \
        --output_root "${intermediate_root_path}" \
        --output_subdir "${intermediate_tag}" \
        --min_prob "${min_prob}" \
        --max_prob "${max_prob}"

    if [[ ! -f "${input_json_path}" ]]; then
        echo "Failed to generate input json: ${input_json_path}"
        exit 1
    fi

    if [[ -n "${limit}" ]]; then
        python3 LLM_TEST/llm_api_judge.py \
            --config "${config_path}" \
            --env_file "${env_path}" \
            --input_json "${input_json_path}" \
            --prompt_file "${prompt_file_path}" \
            --output_root "${output_root_path}" \
            --output_name "${output_tag}" \
            --workers "${workers}" \
            --limit "${limit}"
    else
        python3 LLM_TEST/llm_api_judge.py \
            --config "${config_path}" \
            --env_file "${env_path}" \
            --input_json "${input_json_path}" \
            --prompt_file "${prompt_file_path}" \
            --output_root "${output_root_path}" \
            --output_name "${output_tag}" \
            --workers "${workers}"
    fi

    python3 LLM_TEST/recompute_metrics.py \
        --config "${config_path}" \
        --env_file "${env_path}" \
        --results_csv "${results_csv}" \
        --llm_predictions_csv "${llm_predictions_csv}" \
        --output_dir "${metrics_output_dir}"

    echo "Completed."
}

if [[ $# -gt 0 ]]; then
    case "$1" in
        --result-model|--dataset|--prob-range|--limit|--config|--env|--data-json|--prompt-file|--output-root|-h|--help)
            run_auto_mode "$@"
            exit 0
            ;;
    esac
fi

run_legacy_mode "$@"
