import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize classification, error, and similarity metrics for a standalone analysis run."
    )
    parser.add_argument("--results-csv", required=True, help="Path to original results.csv.")
    parser.add_argument("--errors-csv", required=True, help="Path to collected errors CSV.")
    parser.add_argument(
        "--similarity-csvs",
        nargs="+",
        default=[],
        help="One or more similarity CSV files to summarize.",
    )
    parser.add_argument("--output-json", required=True, help="Path to save summary JSON.")
    parser.add_argument("--output-csv", required=True, help="Path to save summary CSV.")
    return parser.parse_args()


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_int(value, default=0):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def to_float(value, default=0.0):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def classification_metrics(results_rows):
    tp = tn = fp = fn = 0
    for row in results_rows:
        label = to_int(row.get("Label"))
        prediction = to_int(row.get("Prediction"))
        if label == 1 and prediction == 1:
            tp += 1
        elif label == 0 and prediction == 0:
            tn += 1
        elif label == 0 and prediction == 1:
            fp += 1
        elif label == 1 and prediction == 0:
            fn += 1

    total = tp + tn + fp + fn
    accuracy = safe_div(tp + tn, total)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)
    return {
        "count": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "positive_labels": tp + fn,
        "negative_labels": tn + fp,
        "positive_predictions": tp + fp,
        "negative_predictions": tn + fn,
    }


def error_metrics(error_rows):
    fp_rows = [row for row in error_rows if row.get("error_type") == "FP"]
    fn_rows = [row for row in error_rows if row.get("error_type") == "FN"]
    probs = [to_float(row.get("prob"), default=None) for row in error_rows if row.get("prob") not in (None, "")]
    probs = [value for value in probs if value is not None]
    return {
        "error_count": len(error_rows),
        "fp_count": len(fp_rows),
        "fn_count": len(fn_rows),
        "fp_ratio_among_errors": safe_div(len(fp_rows), len(error_rows)),
        "fn_ratio_among_errors": safe_div(len(fn_rows), len(error_rows)),
        "prob_mean": mean(probs) if probs else 0.0,
        "prob_median": median(probs) if probs else 0.0,
        "prob_min": min(probs) if probs else 0.0,
        "prob_max": max(probs) if probs else 0.0,
    }


def summarize_similarity(path: Path):
    rows = load_csv(path)
    similarities = [to_float(row.get("similarity")) for row in rows]
    by_query = defaultdict(list)
    same_error_type = 0
    same_prediction = 0
    neighbor_error_rows = 0

    for row in rows:
        by_query[row["query_index"]].append(to_float(row["similarity"]))
        if row.get("query_error_type") == row.get("neighbor_error_type"):
            same_error_type += 1
        if row.get("query_prediction") == row.get("neighbor_prediction"):
            same_prediction += 1
        if row.get("neighbor_error_type") and row.get("neighbor_error_type") != "correct":
            neighbor_error_rows += 1

    per_query_mean = [mean(values) for values in by_query.values()] if by_query else []
    summary = {
        "file": str(path),
        "pair_count": len(rows),
        "query_count": len(by_query),
        "similarity_mean": mean(similarities) if similarities else 0.0,
        "similarity_median": median(similarities) if similarities else 0.0,
        "similarity_min": min(similarities) if similarities else 0.0,
        "similarity_max": max(similarities) if similarities else 0.0,
        "per_query_similarity_mean": mean(per_query_mean) if per_query_mean else 0.0,
        "same_error_type_ratio": safe_div(same_error_type, len(rows)),
        "same_prediction_ratio": safe_div(same_prediction, len(rows)),
        "neighbor_error_ratio": safe_div(neighbor_error_rows, len(rows)),
    }
    return summary


def flatten_summary(dataset_summary):
    rows = []
    cls = dataset_summary["classification"]
    err = dataset_summary["errors"]
    rows.append({"section": "classification", "metric": key, "value": value} for key, value in cls.items())
    rows.append({"section": "errors", "metric": key, "value": value} for key, value in err.items())
    flat_rows = []
    for group in rows:
        flat_rows.extend(group)
    for item in dataset_summary["similarity"]:
        file_name = Path(item["file"]).name
        for key, value in item.items():
            if key == "file":
                continue
            flat_rows.append({"section": f"similarity:{file_name}", "metric": key, "value": value})
    return flat_rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    results_rows = load_csv(Path(args.results_csv))
    error_rows = load_csv(Path(args.errors_csv))
    similarity_summaries = [summarize_similarity(Path(path)) for path in args.similarity_csvs]

    summary = {
        "results_csv": args.results_csv,
        "errors_csv": args.errors_csv,
        "classification": classification_metrics(results_rows),
        "errors": error_metrics(error_rows),
        "similarity": similarity_summaries,
    }

    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_csv, flatten_summary(summary))

    print(f"results_csv={args.results_csv}")
    print(f"errors_csv={args.errors_csv}")
    print(f"similarity_files={len(similarity_summaries)}")
    print(f"output_json={output_json}")
    print(f"output_csv={output_csv}")


if __name__ == "__main__":
    main()
