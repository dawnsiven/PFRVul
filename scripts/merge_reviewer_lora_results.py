#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge reviewer LoRA predictions back into the original small-model results "
            "and compute final metrics."
        )
    )
    parser.add_argument("--original_results_csv", required=True, help="Path to the original small-model results.csv")
    parser.add_argument("--reviewer_results_csv", required=True, help="Path to the LoRA reviewer results.csv")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for merged outputs. Defaults to the parent directory of --reviewer_results_csv.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def compute_metrics(labels: List[int], preds: List[int]) -> Dict[str, float]:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

    total = len(labels)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }


def main() -> None:
    args = parse_args()
    original_results_csv = Path(args.original_results_csv).resolve()
    reviewer_results_csv = Path(args.reviewer_results_csv).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else reviewer_results_csv.parent.resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    original_rows = load_csv(original_results_csv)
    reviewer_rows = load_csv(reviewer_results_csv)
    reviewer_by_index = {to_int(row.get("Index")): row for row in reviewer_rows}

    merged_rows: List[dict] = []
    labels: List[int] = []
    original_preds: List[int] = []
    final_preds: List[int] = []
    reviewed_labels: List[int] = []
    reviewed_original_preds: List[int] = []
    reviewed_final_preds: List[int] = []

    reviewer_applied_count = 0
    reviewer_changed_count = 0
    reviewer_missing_count = 0

    for row in original_rows:
        index = to_int(row.get("Index"))
        label = to_int(row.get("Label"))
        original_pred = to_int(row.get("Prediction"))
        original_prob = row.get("Prob", row.get("prob", ""))
        reviewer_row = reviewer_by_index.get(index)

        final_pred = original_pred
        decision_source = "small_model_unreviewed_negative"
        reviewer_pred_text = ""
        reviewer_prob_text = ""
        reviewer_response = ""

        if original_pred == 1:
            reviewed_labels.append(label)
            reviewed_original_preds.append(original_pred)

            if reviewer_row and str(reviewer_row.get("Prediction", "")).strip() in {"0", "1"}:
                reviewer_pred_text = str(reviewer_row.get("Prediction", "")).strip()
                reviewer_prob_text = str(reviewer_row.get("Prob", "")).strip()
                reviewer_response = str(reviewer_row.get("Response", "")).strip()
                final_pred = to_int(reviewer_pred_text)
                decision_source = "reviewer_override_positive_subset"
                reviewer_applied_count += 1
                if final_pred != original_pred:
                    reviewer_changed_count += 1
            else:
                decision_source = "small_model_positive_missing_reviewer"
                reviewer_missing_count += 1

            reviewed_final_preds.append(final_pred)

        labels.append(label)
        original_preds.append(original_pred)
        final_preds.append(final_pred)

        merged_rows.append(
            {
                "Index": index,
                "Label": label,
                "OriginalPrediction": original_pred,
                "FinalPrediction": final_pred,
                "OriginalProb": original_prob,
                "ReviewerPrediction": reviewer_pred_text,
                "ReviewerProb": reviewer_prob_text,
                "ReviewerResponse": reviewer_response,
                "DecisionSource": decision_source,
            }
        )

    original_metrics = compute_metrics(labels, original_preds)
    final_metrics = compute_metrics(labels, final_preds)
    reviewed_subset_original_metrics = compute_metrics(reviewed_labels, reviewed_original_preds)
    reviewed_subset_final_metrics = compute_metrics(reviewed_labels, reviewed_final_preds)

    summary = {
        "original_results_csv": str(original_results_csv),
        "reviewer_results_csv": str(reviewer_results_csv),
        "reviewer_rows": len(reviewer_rows),
        "reviewer_applied_count": reviewer_applied_count,
        "reviewer_changed_count": reviewer_changed_count,
        "reviewer_missing_count": reviewer_missing_count,
    }

    merged_csv_path = output_dir / "merged_results.csv"
    with merged_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Index",
                "Label",
                "OriginalPrediction",
                "FinalPrediction",
                "OriginalProb",
                "ReviewerPrediction",
                "ReviewerProb",
                "ReviewerResponse",
                "DecisionSource",
            ],
        )
        writer.writeheader()
        writer.writerows(merged_rows)

    metrics_path = output_dir / "merged_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "original_metrics": original_metrics,
                "final_metrics": final_metrics,
                "reviewed_subset_original_metrics": reviewed_subset_original_metrics,
                "reviewed_subset_final_metrics": reviewed_subset_final_metrics,
                "summary": summary,
                "merged_results_csv": str(merged_csv_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("original_metrics")
    print(json.dumps(original_metrics, ensure_ascii=False, indent=2))
    print("final_metrics")
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    print("reviewed_subset_original_metrics")
    print(json.dumps(reviewed_subset_original_metrics, ensure_ascii=False, indent=2))
    print("reviewed_subset_final_metrics")
    print(json.dumps(reviewed_subset_final_metrics, ensure_ascii=False, indent=2))
    print("summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"merged_results={merged_csv_path}")
    print(f"merged_metrics={metrics_path}")


if __name__ == "__main__":
    main()
