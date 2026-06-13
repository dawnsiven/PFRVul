#!/bin/bash
set -euo pipefail

DATASET_NAME=${1:-}
RESULT_MODEL_NAME=${2:-}
LENGTH=${3:-}
POS_RATIO=""
OOF_SOURCE_HINT=""

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

if [ $# -lt 3 ]; then
    echo "Usage: $0 <DATASET_NAME> <RESULT_MODEL_NAME> <LENGTH> [POS_RATIO] [OOF_RUN_TAG|OOF_TRAIN_CSV]"
    exit 1
fi

if [ $# -eq 4 ]; then
    if [[ "$4" =~ ^[0-9]+$ ]]; then
        POS_RATIO="$4"
    else
        OOF_SOURCE_HINT="$4"
    fi
elif [ $# -ge 5 ]; then
    POS_RATIO="$4"
    OOF_SOURCE_HINT="${5:-}"
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

REVIEWER_DATA_DIR="reviewer_finetune_data"
DATASET_TAG="${DATASET_NAME}_${LENGTH}"
REVIEWER_DATA_SUBDIR="${REPO_ROOT}/${REVIEWER_DATA_DIR}/${RESULT_MODEL_NAME}/${DATASET_TAG}"
REBUCKETED_DATA_SUBDIR="${REPO_ROOT}/${REVIEWER_DATA_DIR}/${RESULT_MODEL_NAME}/${DATASET_TAG}_length_rebucketed"
SOURCE_TRAIN_JSON="${REPO_ROOT}/data/${DATASET_NAME}/alpaca/${DATASET_TAG}_train.json"
OOF_ROOT_DIR="${REPO_ROOT}/outputs/${RESULT_MODEL_NAME}_oof/${DATASET_TAG}"

if [[ -n "${POS_RATIO}" ]]; then
    DATASET_TAG="${DATASET_NAME}_${LENGTH}_${POS_RATIO}"
    REVIEWER_DATA_SUBDIR="${REPO_ROOT}/${REVIEWER_DATA_DIR}/${RESULT_MODEL_NAME}_imbalance/${DATASET_TAG}"
    REBUCKETED_DATA_SUBDIR="${REPO_ROOT}/${REVIEWER_DATA_DIR}/${RESULT_MODEL_NAME}_imbalance/${DATASET_TAG}_length_rebucketed"
    SOURCE_TRAIN_JSON="${REPO_ROOT}/data/${DATASET_NAME}_subsampled/alpaca/${DATASET_TAG}_train.json"
    OOF_ROOT_DIR="${REPO_ROOT}/outputs/${RESULT_MODEL_NAME}_imbalance_oof/${DATASET_TAG}"
fi

resolve_oof_train_csv() {
    local source_hint="$1"

    if [[ -n "${source_hint}" ]]; then
        if [[ -f "${source_hint}" ]]; then
            realpath "${source_hint}"
            return
        fi

        local tagged_csv="${OOF_ROOT_DIR}/${source_hint}/reviewer_train.csv"
        if [[ -f "${tagged_csv}" ]]; then
            realpath "${tagged_csv}"
            return
        fi

        echo ""
        return
    fi

    if [[ ! -d "${OOF_ROOT_DIR}" ]]; then
        echo ""
        return
    fi

    local latest_csv
    latest_csv=$(find "${OOF_ROOT_DIR}" -mindepth 2 -maxdepth 2 -type f -name reviewer_train.csv -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}')
    if [[ -n "${latest_csv}" && -f "${latest_csv}" ]]; then
        realpath "${latest_csv}"
        return
    fi

    echo ""
}

OOF_TRAIN_CSV_ABS=$(resolve_oof_train_csv "${OOF_SOURCE_HINT}")

if [[ -z "${OOF_TRAIN_CSV_ABS}" || ! -f "${OOF_TRAIN_CSV_ABS}" ]]; then
    echo "Could not resolve OOF reviewer_train.csv."
    echo "Searched under: ${OOF_ROOT_DIR}"
    if [[ -n "${OOF_SOURCE_HINT}" ]]; then
        echo "Given hint: ${OOF_SOURCE_HINT}"
        echo "You can pass either:"
        echo "  1. an OOF run tag, e.g. myrun_eval_on_original_val"
        echo "  2. a full path to reviewer_train.csv"
    else
        echo "No OOF run tag or CSV path was provided, so the script tried to auto-detect the latest reviewer_train.csv."
    fi
    exit 1
fi

if [[ ! -f "${SOURCE_TRAIN_JSON}" ]]; then
    echo "Missing source train JSON: ${SOURCE_TRAIN_JSON}"
    exit 1
fi

mkdir -p "${REVIEWER_DATA_SUBDIR}"

echo "Dataset: ${DATASET_NAME}"
echo "Reviewer result model: ${RESULT_MODEL_NAME}"
echo "Length: ${LENGTH}"
if [[ -n "${POS_RATIO}" ]]; then
    echo "POS_RATIO: ${POS_RATIO}"
fi
echo "Using Python interpreter: ${PYTHON_BIN}"
echo "OOF train CSV: ${OOF_TRAIN_CSV_ABS}"
if [[ -n "${OOF_SOURCE_HINT}" ]]; then
    echo "OOF source hint: ${OOF_SOURCE_HINT}"
else
    echo "OOF source hint: <auto-detected latest>"
fi
echo "Reviewer data dir: ${REVIEWER_DATA_SUBDIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/generate_reviewer_finetune_json.py" \
    --dataset-name "${DATASET_NAME}" \
    --result-model "${RESULT_MODEL_NAME}" \
    --length "${LENGTH}" \
    --pos-ratio "${POS_RATIO}" \
    --repo-root "${REPO_ROOT}" \
    --output-root "${REVIEWER_DATA_DIR}"

"${PYTHON_BIN}" - "${OOF_TRAIN_CSV_ABS}" "${SOURCE_TRAIN_JSON}" "${REVIEWER_DATA_SUBDIR}/train.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

INSTRUCTION = (
    "The small model predicts that the following code contains a vulnerability. "
    "Determine whether this prediction should be kept or rejected."
)

csv_path = Path(sys.argv[1])
source_train_json = Path(sys.argv[2])
output_train_json = Path(sys.argv[3])

source_records = json.loads(source_train_json.read_text(encoding="utf-8"))
source_by_index = {}
for record in source_records:
    idx = record.get("index")
    if idx is None:
        raise ValueError(f"Missing index in source record: {record}")
    source_by_index[int(idx)] = record

rows = []
with csv_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    required = {"Index", "Label", "Prediction", "prob"}
    missing = required.difference(reader.fieldnames or [])
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    for row in reader:
        pred = int(row["Prediction"])
        if pred != 1:
            continue

        sample_index = int(row["Index"])
        source = source_by_index.get(sample_index)
        if source is None:
            raise KeyError(f"Index {sample_index} from {csv_path} not found in {source_train_json}")

        prob_text = str(row.get("prob", "")).strip()
        label = int(row["Label"])

        rows.append(
            {
                "instruction": INSTRUCTION,
                "input": f"Small model confidence: {prob_text}\n\nCode:\n{source['input']}",
                "prob": float(prob_text) if prob_text else 0.0,
                "output": "1" if label == 1 else "0",
                "index": sample_index,
            }
        )

output_train_json.parent.mkdir(parents=True, exist_ok=True)
output_train_json.write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(f"Wrote {len(rows)} OOF train samples to {output_train_json}")
PY

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/rebucket_reviewer_json_by_length.py" \
    --input-dir "${REVIEWER_DATA_SUBDIR}"

echo "Prepared reviewer finetune data with:"
echo "  train  = OOF reviewer_train.csv converted from ${OOF_TRAIN_CSV_ABS}"
echo "  val    = original reviewer val.json"
echo "  test   = original reviewer test.json"
echo "Rebucketed data is ready under:"
echo "  ${REBUCKETED_DATA_SUBDIR}"
