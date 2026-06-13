import argparse
import csv
import json
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path("improve_recall/output")


def safe_div(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def compute_metrics(labels, preds):
    tp = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 1)
    tn = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 0)
    fp = sum(1 for label, pred in zip(labels, preds) if label == 0 and pred == 1)
    fn = sum(1 for label, pred in zip(labels, preds) if label == 1 and pred == 0)

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    fpr = safe_div(fp, fp + tn)

    return {
        "count": tp + tn + fp + fn,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "fpr": round(fpr, 6),
    }


def summarize_dataset(dataset_name, output_root):
    result_csv = output_root / dataset_name / "result.csv"
    if not result_csv.exists():
        raise FileNotFoundError(f"Merged result file not found: {result_csv}")

    with result_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    labels = [int(row["Label"]) for row in rows]
    codebert_preds = [int(row["CodeBERT_Prediction"]) for row in rows]
    unixcoder_preds = [int(row["UniXcoder_Prediction"]) for row in rows]
    final_preds = [int(row["Final_Prediction"]) for row in rows]
    total_count = len(rows)

    same_prediction_count = sum(
        1
        for row in rows
        if row["CodeBERT_Prediction"] == row["UniXcoder_Prediction"]
    )
    different_prediction_count = total_count - same_prediction_count
    both_positive_count = sum(
        1
        for row in rows
        if row["CodeBERT_Prediction"] == "1" and row["UniXcoder_Prediction"] == "1"
    )
    both_negative_count = sum(
        1
        for row in rows
        if row["CodeBERT_Prediction"] == "0" and row["UniXcoder_Prediction"] == "0"
    )
    codebert_only_positive_count = sum(
        1
        for row in rows
        if row["CodeBERT_Prediction"] == "1" and row["UniXcoder_Prediction"] == "0"
    )
    unixcoder_only_positive_count = sum(
        1
        for row in rows
        if row["CodeBERT_Prediction"] == "0" and row["UniXcoder_Prediction"] == "1"
    )

    codebert_positive_indices = {
        row["Index"] for row in rows if row["CodeBERT_Prediction"] == "1"
    }
    unixcoder_positive_indices = {
        row["Index"] for row in rows if row["UniXcoder_Prediction"] == "1"
    }
    positive_intersection_count = len(codebert_positive_indices & unixcoder_positive_indices)
    positive_union_count = len(codebert_positive_indices | unixcoder_positive_indices)

    summary = {
        "dataset": dataset_name,
        "strategy": rows[0]["Strategy"] if rows else "unknown",
        "positive_samples": sum(labels),
        "negative_samples": len(labels) - sum(labels),
        "codebert": compute_metrics(labels, codebert_preds),
        "unixcoder": compute_metrics(labels, unixcoder_preds),
        "final": compute_metrics(labels, final_preds),
        "overlap": {
            "same_prediction_count": same_prediction_count,
            "same_prediction_ratio": round(same_prediction_count / total_count, 6) if total_count else 0.0,
            "different_prediction_count": different_prediction_count,
            "different_prediction_ratio": round(different_prediction_count / total_count, 6) if total_count else 0.0,
            "both_positive_count": both_positive_count,
            "both_negative_count": both_negative_count,
            "codebert_only_positive_count": codebert_only_positive_count,
            "unixcoder_only_positive_count": unixcoder_only_positive_count,
            "positive_intersection_count": positive_intersection_count,
            "positive_union_count": positive_union_count,
            "positive_jaccard": round(
                positive_intersection_count / positive_union_count, 6
            ) if positive_union_count else 0.0,
        },
        "disagreement_count": sum(
            1
            for row in rows
            if row["CodeBERT_Prediction"] != row["UniXcoder_Prediction"]
        ),
        "rescued_positives": sum(
            1
            for row in rows
            if int(row["Label"]) == 1
            and int(row["CodeBERT_Prediction"]) == 0
            and int(row["UniXcoder_Prediction"]) == 1
            and int(row["Final_Prediction"]) == 1
        )
        + sum(
            1
            for row in rows
            if int(row["Label"]) == 1
            and int(row["CodeBERT_Prediction"]) == 1
            and int(row["UniXcoder_Prediction"]) == 0
            and int(row["Final_Prediction"]) == 1
        ),
    }

    summary_path = output_root / dataset_name / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    summary_csv_path = output_root / dataset_name / "summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "CodeBERT", "UniXcoder", "Final"],
        )
        writer.writeheader()
        for metric in ["count", "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1", "fpr"]:
            writer.writerow(
                {
                    "metric": metric,
                    "CodeBERT": summary["codebert"][metric],
                    "UniXcoder": summary["unixcoder"][metric],
                    "Final": summary["final"][metric],
                }
            )
        writer.writerow(
            {
                "metric": "same_prediction_ratio",
                "CodeBERT": "",
                "UniXcoder": "",
                "Final": summary["overlap"]["same_prediction_ratio"],
            }
        )
        writer.writerow(
            {
                "metric": "different_prediction_ratio",
                "CodeBERT": "",
                "UniXcoder": "",
                "Final": summary["overlap"]["different_prediction_ratio"],
            }
        )
        writer.writerow(
            {
                "metric": "positive_jaccard",
                "CodeBERT": "",
                "UniXcoder": "",
                "Final": summary["overlap"]["positive_jaccard"],
            }
        )

    return result_csv, summary_path, summary_csv_path, summary


def build_parser():
    parser = argparse.ArgumentParser(
        description="Summarize metrics for merged improve_recall results."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset folder name, for example: bigvul_cwe20_1_1",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory of merged outputs. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    _, summary_path, summary_csv_path, summary = summarize_dataset(
        dataset_name=args.dataset,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to: {summary_path}")
    print(f"Summary CSV saved to: {summary_csv_path}")


if __name__ == "__main__":
    main()
