import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def load_rows(csv_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"Label", "Prediction", "prob"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")

        for row in reader:
            label = int(row["Label"])
            pred = int(row["Prediction"])
            prob = float(row["prob"])
            confidence = max(prob, 1.0 - prob)
            correct = 1 if label == pred else 0
            rows.append(
                {
                    "label": label,
                    "prediction": pred,
                    "prob": prob,
                    "confidence": confidence,
                    "correct": correct,
                }
            )
    if not rows:
        raise ValueError("CSV 中没有有效样本")
    return rows


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def summarize_overall(rows: List[Dict[str, float]]) -> Dict[str, float]:
    total = len(rows)
    accuracy = safe_div(sum(row["correct"] for row in rows), total)
    avg_confidence = safe_div(sum(row["confidence"] for row in rows), total)
    return {
        "total_samples": total,
        "overall_accuracy": accuracy,
        "average_confidence": avg_confidence,
        "confidence_accuracy_gap": avg_confidence - accuracy,
        "min_confidence": min(row["confidence"] for row in rows),
        "max_confidence": max(row["confidence"] for row in rows),
    }


def build_bins(rows: List[Dict[str, float]], bin_width: float) -> List[Dict[str, float]]:
    bins: List[Dict[str, float]] = []
    start = 0.5
    while start < 1.0:
        end = min(start + bin_width, 1.0)
        subset = [
            row for row in rows
            if start <= row["confidence"] < end
            or (end >= 1.0 and start <= row["confidence"] <= 1.0)
        ]
        if subset:
            count = len(subset)
            accuracy = safe_div(sum(row["correct"] for row in subset), count)
            avg_conf = safe_div(sum(row["confidence"] for row in subset), count)
            bins.append(
                {
                    "bin_start": round(start, 4),
                    "bin_end": round(end, 4),
                    "count": count,
                    "ratio": safe_div(count, len(rows)),
                    "accuracy": accuracy,
                    "avg_confidence": avg_conf,
                    "gap": avg_conf - accuracy,
                }
            )
        start = round(start + bin_width, 10)
    return bins


def build_thresholds(rows: List[Dict[str, float]], thresholds: List[float]) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    total = len(rows)
    for threshold in thresholds:
        subset = [row for row in rows if row["confidence"] >= threshold]
        count = len(subset)
        results.append(
            {
                "threshold": threshold,
                "count": count,
                "coverage": safe_div(count, total),
                "accuracy": safe_div(sum(row["correct"] for row in subset), count),
                "avg_confidence": safe_div(sum(row["confidence"] for row in subset), count),
            }
        )
    return results


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="统计置信度与准确率之间的关系")
    parser.add_argument("--input_csv", required=True, help="模型预测结果 CSV，需包含 Label/Prediction/prob 列")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--bin_width", type=float, default=0.05, help="置信度分桶宽度，默认 0.05")
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95],
        help="统计高于这些置信度阈值时的覆盖率和准确率",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_csv)
    overall = summarize_overall(rows)
    bins = build_bins(rows, args.bin_width)
    thresholds = build_thresholds(rows, args.thresholds)

    write_csv(output_dir / "confidence_bins.csv", bins)
    write_csv(output_dir / "confidence_thresholds.csv", thresholds)

    summary = {
        "input_csv": str(input_csv),
        "confidence_definition": "max(prob, 1-prob), 其中 prob 是正类概率，Prediction 由 prob > 0.5 得到",
        "overall": overall,
        "bins": bins,
        "thresholds": thresholds,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("overall")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("top_thresholds")
    print(json.dumps(thresholds, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
