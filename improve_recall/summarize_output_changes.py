import argparse
import csv
import json
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path("improve_recall/output")
DEFAULT_SUMMARY_DIR = Path("improve_recall/output_summary")
METRICS = ["accuracy", "precision", "recall", "f1", "fpr"]


def load_summary(summary_path):
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_type_name(dataset_name):
    parts = dataset_name.split("_")
    if len(parts) >= 2:
        return parts[1]
    return dataset_name


def build_dataset_row(summary):
    dataset = summary["dataset"]
    row = {
        "dataset": dataset,
        "type": extract_type_name(dataset),
        "strategy": summary.get("strategy", "unknown"),
        "positive_samples": summary.get("positive_samples", 0),
        "negative_samples": summary.get("negative_samples", 0),
        "same_prediction_ratio": summary.get("overlap", {}).get("same_prediction_ratio", 0.0),
        "different_prediction_ratio": summary.get("overlap", {}).get("different_prediction_ratio", 0.0),
        "positive_jaccard": summary.get("overlap", {}).get("positive_jaccard", 0.0),
        "rescued_positives": summary.get("rescued_positives", 0),
    }

    codebert = summary["codebert"]
    unixcoder = summary["unixcoder"]
    final = summary["final"]

    for metric in METRICS:
        row[f"codebert_{metric}"] = codebert[metric]
        row[f"unixcoder_{metric}"] = unixcoder[metric]
        row[f"final_{metric}"] = final[metric]
        row[f"delta_final_vs_codebert_{metric}"] = round(final[metric] - codebert[metric], 6)
        row[f"delta_final_vs_unixcoder_{metric}"] = round(final[metric] - unixcoder[metric], 6)

    best_single_recall = max(codebert["recall"], unixcoder["recall"])
    best_single_f1 = max(codebert["f1"], unixcoder["f1"])
    best_single_precision = max(codebert["precision"], unixcoder["precision"])
    best_single_accuracy = max(codebert["accuracy"], unixcoder["accuracy"])
    best_single_fpr = min(codebert["fpr"], unixcoder["fpr"])

    row["delta_final_vs_best_recall"] = round(final["recall"] - best_single_recall, 6)
    row["delta_final_vs_best_f1"] = round(final["f1"] - best_single_f1, 6)
    row["delta_final_vs_best_precision"] = round(final["precision"] - best_single_precision, 6)
    row["delta_final_vs_best_accuracy"] = round(final["accuracy"] - best_single_accuracy, 6)
    row["delta_final_vs_best_fpr"] = round(final["fpr"] - best_single_fpr, 6)
    return row


def mean(values):
    return round(sum(values) / len(values), 6) if values else 0.0


def build_overall_summary(rows):
    summary = {
        "dataset_count": len(rows),
        "datasets": [row["dataset"] for row in rows],
        "avg_same_prediction_ratio": mean([row["same_prediction_ratio"] for row in rows]),
        "avg_positive_jaccard": mean([row["positive_jaccard"] for row in rows]),
        "total_rescued_positives": sum(row["rescued_positives"] for row in rows),
        "metrics": {},
    }

    for metric in METRICS:
        summary["metrics"][metric] = {
            "avg_codebert": mean([row[f"codebert_{metric}"] for row in rows]),
            "avg_unixcoder": mean([row[f"unixcoder_{metric}"] for row in rows]),
            "avg_final": mean([row[f"final_{metric}"] for row in rows]),
            "avg_delta_final_vs_codebert": mean(
                [row[f"delta_final_vs_codebert_{metric}"] for row in rows]
            ),
            "avg_delta_final_vs_unixcoder": mean(
                [row[f"delta_final_vs_unixcoder_{metric}"] for row in rows]
            ),
        }

    return summary


def write_csv(rows, csv_path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_output_changes(output_root, summary_dir):
    summary_files = sorted(output_root.glob("*/summary.json"))
    if not summary_files:
        raise FileNotFoundError(f"No summary.json files found under: {output_root}")

    rows = [build_dataset_row(load_summary(path)) for path in summary_files]
    rows.sort(key=lambda item: item["dataset"])

    summary_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv_path = summary_dir / "dataset_changes.csv"
    dataset_json_path = summary_dir / "dataset_changes.json"
    overall_json_path = summary_dir / "overall_changes.json"

    write_csv(rows, dataset_csv_path)
    with dataset_json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)

    overall_summary = build_overall_summary(rows)
    with overall_json_path.open("w", encoding="utf-8") as handle:
        json.dump(overall_summary, handle, indent=2, ensure_ascii=False)

    return dataset_csv_path, dataset_json_path, overall_json_path, overall_summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize metric changes for all datasets under improve_recall/output."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory containing per-dataset outputs. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=DEFAULT_SUMMARY_DIR,
        help=f"Directory to save aggregated summaries. Default: {DEFAULT_SUMMARY_DIR}",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    dataset_csv_path, dataset_json_path, overall_json_path, overall_summary = summarize_output_changes(
        output_root=args.output_root,
        summary_dir=args.summary_dir,
    )
    print(json.dumps(overall_summary, indent=2, ensure_ascii=False))
    print(f"Dataset CSV saved to: {dataset_csv_path}")
    print(f"Dataset JSON saved to: {dataset_json_path}")
    print(f"Overall JSON saved to: {overall_json_path}")


if __name__ == "__main__":
    main()
