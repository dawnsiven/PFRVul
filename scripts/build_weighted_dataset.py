import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_default_weight(path, weight_key, default_weight):
    data = load_json(path)
    changed = False
    for item in data:
        if weight_key not in item:
            item[weight_key] = default_weight
            changed = True
    if changed:
        dump_json(path, data)
    return len(data), changed


def normalize_row_key(row, *candidates):
    lowered = {key.lower(): key for key in row}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None:
            return row[key]
    raise KeyError(f"Missing columns {candidates} in csv row with columns {list(row.keys())}")


def load_error_indices(result_csv):
    wrong_indices = set()
    with open(result_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = int(normalize_row_key(row, "Index"))
            label = int(normalize_row_key(row, "Label"))
            prediction = int(normalize_row_key(row, "Prediction"))
            if label != prediction:
                wrong_indices.add(index)
    return wrong_indices


def update_weight(current_weight, error_weight, mode):
    if mode == "set":
        return error_weight
    if mode == "add":
        return current_weight + error_weight
    if mode == "multiply":
        return current_weight * error_weight
    raise ValueError(f"Unsupported mode: {mode}")


def build_weighted_train(train_path, output_path, wrong_indices, weight_key, index_key, error_weight, mode):
    data = load_json(train_path)
    updated = 0
    missing_index = 0
    for item in data:
        current_weight = float(item.get(weight_key, 1.0))
        item[weight_key] = current_weight
        sample_index = item.get(index_key)
        if sample_index is None:
            missing_index += 1
            continue
        if int(sample_index) in wrong_indices:
            item[weight_key] = update_weight(current_weight, error_weight, mode)
            updated += 1

    dump_json(output_path, data)
    return {
        "train_size": len(data),
        "updated_samples": updated,
        "missing_index_samples": missing_index,
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Add default weights and build a weighted train file from model errors.")
    parser.add_argument(
        "--dataset-dir",
        default="data/bigvul_cwe20_subsampled/alpaca",
        help="Directory containing the dataset json files to backfill with default weights.",
    )
    parser.add_argument(
        "--dataset-pattern",
        default="bigvul_cwe20_*.json",
        help="Glob pattern for dataset files inside dataset-dir.",
    )
    parser.add_argument(
        "--train-file",
        default="data/bigvul_cwe20_subsampled/alpaca/bigvul_cwe20_1_1_train.json",
        help="Train json file to use as the source for the new weighted train file.",
    )
    parser.add_argument(
        "--result-csv",
        default="outputs/CodeBERT_imbalance_test/bigvul_cwe20_1_1/results.csv",
        help="CSV file containing Index/Label/Prediction columns.",
    )
    parser.add_argument(
        "--output-train-file",
        default="data/bigvul_cwe20_subsampled/alpaca/bigvul_cwe20_1_1_train_weighted.json",
        help="Output path for the generated weighted train json file.",
    )
    parser.add_argument("--weight-key", default="weight", help="Weight field name in the json data.")
    parser.add_argument("--index-key", default="index", help="Index field name in the json data.")
    parser.add_argument("--default-weight", type=float, default=1.0, help="Default weight for samples without weight.")
    parser.add_argument("--error-weight", type=float, default=2.0, help="Weight update value for wrong predictions.")
    parser.add_argument(
        "--update-mode",
        choices=["set", "add", "multiply"],
        default="set",
        help="How to update the weight of incorrectly predicted samples.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    touched = []
    for path in sorted(dataset_dir.glob(args.dataset_pattern)):
        count, changed = ensure_default_weight(path, args.weight_key, args.default_weight)
        touched.append({"path": str(path), "count": count, "changed": changed})

    wrong_indices = load_error_indices(args.result_csv)
    summary = build_weighted_train(
        train_path=args.train_file,
        output_path=args.output_train_file,
        wrong_indices=wrong_indices,
        weight_key=args.weight_key,
        index_key=args.index_key,
        error_weight=args.error_weight,
        mode=args.update_mode,
    )

    print(json.dumps({
        "default_weight_files": touched,
        "wrong_prediction_count": len(wrong_indices),
        "weighted_train_summary": summary,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
