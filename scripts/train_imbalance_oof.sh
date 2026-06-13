#!/bin/bash
set -euo pipefail

DATASET_NAME=${1:-}
MODEL_NAME=${2:-}
LENGTH=${3:-}
POS_RATIO=""
CUDA="0"

if [ $# -lt 3 ]; then
    echo "Usage: $0 <DATASET_NAME> <MODEL_NAME> <LENGTH> [POS_RATIO] [CUDA]"
    echo "Env:"
    echo "  OOF_FOLDS   Number of folds. Default: 5"
    echo "  OOF_SEED    Random seed for fold split. Default: 42"
    echo "  OOF_RUN_TAG Output subdir tag. Default: fold\${OOF_FOLDS}_seed\${OOF_SEED}"
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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
LOCAL_MODEL_ROOT="${REPO_ROOT}/model"

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

OOF_FOLDS=${OOF_FOLDS:-5}
OOF_SEED=${OOF_SEED:-42}
OOF_RUN_TAG=${OOF_RUN_TAG:-"fold${OOF_FOLDS}_seed${OOF_SEED}"}

BLOCK_SIZE=$(echo "$LENGTH" | awk -F'-' '{print $2}')
if [[ -z "$BLOCK_SIZE" ]]; then
    BLOCK_SIZE=512
fi

DATASET_TAG="${DATASET_NAME}_${LENGTH}"
SOURCE_DATA_DIR="${REPO_ROOT}/data/${DATASET_NAME}/alpaca"
ORIGINAL_TRAIN_DATA_FILE="${SOURCE_DATA_DIR}/${DATASET_TAG}_train.json"
ORIGINAL_EVAL_DATA_FILE="${SOURCE_DATA_DIR}/${DATASET_TAG}_validate.json"
OUTPUT_ROOT="${REPO_ROOT}/outputs/${MODEL_NAME}_oof"

if [[ -n "${POS_RATIO}" ]]; then
    DATASET_TAG="${DATASET_NAME}_${LENGTH}_${POS_RATIO}"
    SOURCE_DATA_DIR="${REPO_ROOT}/data/${DATASET_NAME}_subsampled/alpaca"
    ORIGINAL_TRAIN_DATA_FILE="${SOURCE_DATA_DIR}/${DATASET_TAG}_train.json"
    ORIGINAL_EVAL_DATA_FILE="${SOURCE_DATA_DIR}/${DATASET_TAG}_validate.json"
    OUTPUT_ROOT="${REPO_ROOT}/outputs/${MODEL_NAME}_imbalance_oof"
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET_TAG}/${OOF_RUN_TAG}"
WORK_DIR="${OUTPUT_DIR}/oof_work"
OUTPUT_CSV="${OUTPUT_DIR}/reviewer_train.csv"
SUMMARY_JSON="${OUTPUT_DIR}/oof_summary.json"

if [[ ! -f "${ORIGINAL_TRAIN_DATA_FILE}" ]]; then
    echo "Missing train JSON: ${ORIGINAL_TRAIN_DATA_FILE}"
    exit 1
fi

if [[ ! -f "${ORIGINAL_EVAL_DATA_FILE}" ]]; then
    echo "Missing validate JSON: ${ORIGINAL_EVAL_DATA_FILE}"
    exit 1
fi

if [[ -f "${OUTPUT_CSV}" && -f "${SUMMARY_JSON}" ]]; then
    echo "OOF outputs already exist, skipping regeneration:"
    echo "  ${OUTPUT_CSV}"
    echo "  ${SUMMARY_JSON}"
    exit 0
fi

if [[ -e "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Output directory already exists but is incomplete:"
    echo "  ${OUTPUT_DIR}"
    echo "Missing required files:"
    [[ -f "${OUTPUT_CSV}" ]] || echo "  ${OUTPUT_CSV}"
    [[ -f "${SUMMARY_JSON}" ]] || echo "  ${SUMMARY_JSON}"
    echo "Please set a different OOF_RUN_TAG or remove the incomplete directory first."
    exit 1
fi

mkdir -p "${WORK_DIR}"

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

BACKBONE_SOURCE=$(resolve_model_source "microsoft/codebert-base")
TOKENIZER_SOURCE="$BACKBONE_SOURCE"

echo "Dataset tag: ${DATASET_TAG}"
echo "Model: ${MODEL_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA}"
echo "OOF folds: ${OOF_FOLDS}"
echo "OOF seed: ${OOF_SEED}"
if [[ -n "${POS_RATIO}" ]]; then
    echo "POS_RATIO: ${POS_RATIO}"
fi
echo "Using Python interpreter: ${PYTHON_BIN}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Model selection eval JSON: ${ORIGINAL_EVAL_DATA_FILE}"

if [[ "$BACKBONE_SOURCE" == "microsoft/codebert-base" ]]; then
    echo "Using remote backbone: $BACKBONE_SOURCE"
else
    echo "Using local backbone: $BACKBONE_SOURCE"
fi

"${PYTHON_BIN}" - "${ORIGINAL_TRAIN_DATA_FILE}" "${WORK_DIR}" "${OOF_FOLDS}" "${OOF_SEED}" <<'PY'
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

train_path = Path(sys.argv[1])
work_dir = Path(sys.argv[2])
n_folds = int(sys.argv[3])
seed = int(sys.argv[4])

records = json.loads(train_path.read_text(encoding="utf-8"))
if len(records) < n_folds:
    raise ValueError(f"Training samples ({len(records)}) are fewer than folds ({n_folds})")

for pos, record in enumerate(records):
    record["_oof_order"] = pos
    record["_oof_index"] = record.get("index", pos)

buckets = defaultdict(list)
for record in records:
    buckets[str(record.get("output"))].append(record)

rng = random.Random(seed)
folds = [[] for _ in range(n_folds)]
for _, bucket in sorted(buckets.items(), key=lambda item: item[0]):
    rng.shuffle(bucket)
    for i, record in enumerate(bucket):
        folds[i % n_folds].append(record)

manifest = {
    "source_train_json": str(train_path),
    "folds": n_folds,
    "seed": seed,
    "total_samples": len(records),
    "fold_sizes": [len(fold) for fold in folds],
}

for fold_id, valid_records in enumerate(folds):
    train_records = []
    for other_id, other_fold in enumerate(folds):
        if other_id != fold_id:
            train_records.extend(other_fold)

    valid_records = sorted(valid_records, key=lambda item: item["_oof_order"])
    train_records = sorted(train_records, key=lambda item: item["_oof_order"])

    fold_dir = work_dir / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    (fold_dir / "train.json").write_text(
        json.dumps(train_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (fold_dir / "validate.json").write_text(
        json.dumps(valid_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (fold_dir / "meta.json").write_text(
        json.dumps(
            {
                "fold_id": fold_id,
                "train_size": len(train_records),
                "valid_size": len(valid_records),
                "valid_samples": [
                    {"order": item["_oof_order"], "index": item["_oof_index"]}
                    for item in valid_records
                ],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

(work_dir / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

run_fold() {
    local fold_id="$1"
    local fold_dir="${WORK_DIR}/fold_${fold_id}"
    local fold_output_dir="${fold_dir}/model_output"
    local fold_csv_path="${fold_dir}/reviewer_valid.csv"
    local fold_log_path="${fold_dir}/train_${MODEL_NAME}_fold${fold_id}.log"

    mkdir -p "${fold_output_dir}"

    echo "========== Fold ${fold_id}/${OOF_FOLDS} =========="
    echo "Train JSON: ${fold_dir}/train.json"
    echo "OOF target JSON: ${fold_dir}/validate.json"
    echo "Model-selection eval JSON: ${ORIGINAL_EVAL_DATA_FILE}"

    if [[ "$MODEL_NAME" == "GraphCodeBERT" ]]; then
        CUDA_VISIBLE_DEVICES="${CUDA}" "${PYTHON_BIN}" "${MODEL_NAME}/run.py" \
            --output_dir="${fold_output_dir}/" \
            --csv_path="${fold_csv_path}" \
            --tokenizer_name="${TOKENIZER_SOURCE}" \
            --model_name_or_path="${BACKBONE_SOURCE}" \
            --do_train \
            --do_eval \
            --do_test \
            --train_data_file="${fold_dir}/train.json" \
            --eval_data_file="${ORIGINAL_EVAL_DATA_FILE}" \
            --test_data_file="${fold_dir}/validate.json" \
            --epochs 5 \
            --block_size "${BLOCK_SIZE}" \
            --eval_batch_size 4 \
            --train_batch_size 4 \
            --evaluate_during_training \
            --seed "${OOF_SEED}" \
            2>"${fold_log_path}"
    elif [[ "$MODEL_NAME" == "UniXcoder" ]]; then
        CUDA_VISIBLE_DEVICES="${CUDA}" "${PYTHON_BIN}" "${MODEL_NAME}/run.py" \
            --output_dir="${fold_output_dir}/" \
            --csv_path="${fold_csv_path}" \
            --model_name_or_path="${BACKBONE_SOURCE}" \
            --do_train \
            --do_eval \
            --do_test \
            --train_data_file="${fold_dir}/train.json" \
            --eval_data_file="${ORIGINAL_EVAL_DATA_FILE}" \
            --test_data_file="${fold_dir}/validate.json" \
            --num_train_epochs 5 \
            --block_size "${BLOCK_SIZE}" \
            --eval_batch_size 4 \
            --train_batch_size 4 \
            --seed "${OOF_SEED}" \
            2>"${fold_log_path}"
    elif [[ "$MODEL_NAME" == "ReGVD" ]]; then
        CUDA_VISIBLE_DEVICES="${CUDA}" "${PYTHON_BIN}" "${MODEL_NAME}/run.py" \
            --output_dir="${fold_output_dir}/" \
            --csv_path="${fold_csv_path}" \
            --model_type=roberta \
            --tokenizer_name="${TOKENIZER_SOURCE}" \
            --model_name_or_path="${BACKBONE_SOURCE}" \
            --do_train \
            --do_eval \
            --do_test \
            --train_data_file="${fold_dir}/train.json" \
            --eval_data_file="${ORIGINAL_EVAL_DATA_FILE}" \
            --test_data_file="${fold_dir}/validate.json" \
            --num_train_epochs 5 \
            --block_size "${BLOCK_SIZE}" \
            --eval_batch_size 4 \
            --train_batch_size 4 \
            --evaluate_during_training \
            --seed "${OOF_SEED}" \
            2>"${fold_log_path}"
    else
        CUDA_VISIBLE_DEVICES="${CUDA}" "${PYTHON_BIN}" "${MODEL_NAME}/run.py" \
            --output_dir="${fold_output_dir}/" \
            --csv_path="${fold_csv_path}" \
            --model_type=roberta \
            --tokenizer_name="${TOKENIZER_SOURCE}" \
            --model_name_or_path="${BACKBONE_SOURCE}" \
            --do_train \
            --do_eval \
            --do_test \
            --train_data_file="${fold_dir}/train.json" \
            --eval_data_file="${ORIGINAL_EVAL_DATA_FILE}" \
            --test_data_file="${fold_dir}/validate.json" \
            --epoch 5 \
            --block_size "${BLOCK_SIZE}" \
            --eval_batch_size 4 \
            --train_batch_size 4 \
            --evaluate_during_training \
            --seed "${OOF_SEED}" \
            2>"${fold_log_path}"
    fi
}

for (( fold_id=0; fold_id<OOF_FOLDS; fold_id++ )); do
    run_fold "${fold_id}"
done

"${PYTHON_BIN}" - "${WORK_DIR}" "${OUTPUT_CSV}" "${SUMMARY_JSON}" <<'PY'
import csv
import json
import sys
from pathlib import Path

work_dir = Path(sys.argv[1])
output_csv = Path(sys.argv[2])
summary_json = Path(sys.argv[3])

manifest = json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))
all_rows = []
seen_orders = set()
all_extra_keys = set()

for fold_id in range(int(manifest["folds"])):
    fold_dir = work_dir / f"fold_{fold_id}"
    meta = json.loads((fold_dir / "meta.json").read_text(encoding="utf-8"))
    valid_samples = meta["valid_samples"]
    csv_path = fold_dir / "reviewer_valid.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing fold CSV: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != len(valid_samples):
        raise ValueError(
            f"Fold {fold_id} row count mismatch: csv={len(rows)} valid_samples={len(valid_samples)}"
        )

    for sample_meta, row in zip(valid_samples, rows):
        normalized = dict(row)
        normalized["Index"] = str(
            normalized.get("Index", normalized.get("index", sample_meta["index"]))
        )
        normalized["Label"] = str(normalized.get("Label", ""))
        normalized["Prediction"] = str(normalized.get("Prediction", ""))
        if "prob" not in normalized:
            normalized["prob"] = normalized.get("Prob", "")
        normalized["OOFFold"] = str(fold_id)
        normalized["SourceOrder"] = str(sample_meta["order"])

        order = int(sample_meta["order"])
        if order in seen_orders:
            raise ValueError(f"Duplicate OOF order detected: {order}")
        seen_orders.add(order)

        all_rows.append((order, normalized))
        all_extra_keys.update(
            key for key in normalized.keys()
            if key not in {"Index", "Label", "Prediction", "prob", "OOFFold", "SourceOrder"}
        )

expected_total = int(manifest["total_samples"])
if len(all_rows) != expected_total:
    raise ValueError(f"Expected {expected_total} merged rows, got {len(all_rows)}")

all_rows.sort(key=lambda item: item[0])
fieldnames = ["Index", "Label", "Prediction", "prob", "OOFFold", "SourceOrder"] + sorted(all_extra_keys)

output_csv.parent.mkdir(parents=True, exist_ok=True)
with output_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for _, row in all_rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})

summary = {
    "source_train_json": manifest["source_train_json"],
    "folds": manifest["folds"],
    "seed": manifest["seed"],
    "total_samples": manifest["total_samples"],
    "fold_sizes": manifest["fold_sizes"],
    "output_csv": str(output_csv),
    "note": "Each row is predicted by a model that did not train on that sample.",
}
summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo "OOF reviewer train CSV written to:"
echo "  ${OUTPUT_CSV}"
echo "Summary written to:"
echo "  ${SUMMARY_JSON}"
