import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect misclassified samples by joining results.csv with the source test JSON."
    )
    parser.add_argument("--results-csv", required=True, help="Path to model results.csv.")
    parser.add_argument("--data-json", required=True, help="Path to source *_test.json.")
    parser.add_argument("--output-csv", required=True, help="Path to save the joined error CSV.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the joined error records as JSON.",
    )
    return parser.parse_args()


def load_results(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_key(row, *candidates):
    lowered = {key.lower(): key for key in row.keys()}
    for candidate in candidates:
        key = lowered.get(candidate.lower())
        if key is not None:
            return row[key]
    raise KeyError(f"Missing any of columns {candidates}; got {list(row.keys())}")


def to_int(value, default=0):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def build_index_map(data_rows):
    index_map = {}
    for row in data_rows:
        if "index" in row and row["index"] is not None:
            index_map[to_int(row["index"])] = row
    return index_map


def classify_error(label: int, prediction: int) -> str:
    if label == prediction:
        return "correct"
    if label == 0 and prediction == 1:
        return "FP"
    if label == 1 and prediction == 0:
        return "FN"
    return "unknown"


def main():
    args = parse_args()
    results_csv = Path(args.results_csv)
    data_json = Path(args.data_json)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json) if args.output_json else None

    result_rows = load_results(results_csv)
    data_rows = load_json(data_json)
    data_by_index = build_index_map(data_rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Index",
        "Label",
        "Prediction",
        "prob",
        "error_type",
        "matched_by",
        "instruction",
        "input",
        "output",
        "weight",
    ]

    collected = []
    for row in result_rows:
        index = to_int(normalize_key(row, "Index"))
        label = to_int(normalize_key(row, "Label"))
        prediction = to_int(normalize_key(row, "Prediction"))
        if label == prediction:
            continue

        sample = data_by_index.get(index)
        matched_by = "record_index"
        if sample is None and 0 <= index < len(data_rows):
            sample = data_rows[index]
            matched_by = "row_position"

        if sample is None:
            sample = {}
            matched_by = "unmatched"

        prob = None
        for prob_key in ("prob", "Prob"):
            if prob_key in row:
                prob = row[prob_key]
                break

        collected.append(
            {
                "Index": index,
                "Label": label,
                "Prediction": prediction,
                "prob": prob,
                "error_type": classify_error(label, prediction),
                "matched_by": matched_by,
                "instruction": sample.get("instruction", ""),
                "input": sample.get("input", ""),
                "output": sample.get("output", ""),
                "weight": sample.get("weight", ""),
            }
        )

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(collected)

    if output_json:
        with output_json.open("w", encoding="utf-8") as handle:
            json.dump(collected, handle, ensure_ascii=False, indent=2)

    fp_count = sum(1 for row in collected if row["error_type"] == "FP")
    fn_count = sum(1 for row in collected if row["error_type"] == "FN")
    print(f"results_csv={results_csv}")
    print(f"data_json={data_json}")
    print(f"errors={len(collected)}")
    print(f"fp={fp_count}")
    print(f"fn={fn_count}")
    print(f"output_csv={output_csv}")
    if output_json:
        print(f"output_json={output_json}")


if __name__ == "__main__":
    main()
