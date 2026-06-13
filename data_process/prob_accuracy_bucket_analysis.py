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
            prediction = int(row["Prediction"])
            prob = float(row["prob"])
            rows.append(
                {
                    "label": label,
                    "prediction": prediction,
                    "prob": prob,
                    "correct": 1 if label == prediction else 0,
                }
            )

    if not rows:
        raise ValueError("CSV 中没有有效样本")
    return rows


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def build_bin_edges(bin_width: float) -> List[float]:
    if bin_width <= 0 or bin_width > 1:
        raise ValueError("bin_width 必须在 (0, 1] 区间内")

    edges = [0.0]
    current = 0.0
    while current < 1.0:
        current = round(min(current + bin_width, 1.0), 10)
        if current == edges[-1]:
            break
        edges.append(current)
    if edges[-1] != 1.0:
        edges.append(1.0)
    return edges


def build_bins(rows: List[Dict[str, float]], bin_width: float) -> List[Dict[str, float]]:
    edges = build_bin_edges(bin_width)
    bins: List[Dict[str, float]] = []

    for idx in range(len(edges) - 1):
        start = edges[idx]
        end = edges[idx + 1]
        include_right = idx == len(edges) - 2
        subset = []
        for row in rows:
            prob = row["prob"]
            in_bin = start <= prob <= end if include_right else start <= prob < end
            if in_bin:
                subset.append(row)

        if not subset:
            continue

        count = len(subset)
        correct = sum(row["correct"] for row in subset)
        bins.append(
            {
                "bin_start": round(start, 4),
                "bin_end": round(end, 4),
                "count": count,
                "ratio": safe_div(count, len(rows)),
                "accuracy": safe_div(correct, count),
                "avg_prob": safe_div(sum(row["prob"] for row in subset), count),
                "label_1_rate": safe_div(sum(row["label"] for row in subset), count),
                "pred_1_rate": safe_div(sum(row["prediction"] for row in subset), count),
            }
        )

    return bins


def summarize_overall(rows: List[Dict[str, float]]) -> Dict[str, float]:
    total = len(rows)
    return {
        "total_samples": total,
        "overall_accuracy": safe_div(sum(row["correct"] for row in rows), total),
        "average_prob": safe_div(sum(row["prob"] for row in rows), total),
        "min_prob": min(row["prob"] for row in rows),
        "max_prob": max(row["prob"] for row in rows),
    }


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="统计原始 prob 与正确率之间的关系")
    parser.add_argument("--input_csv", required=True, help="模型预测结果 CSV，需包含 Label/Prediction/prob 列")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--bin_width", type=float, default=0.1, help="prob 分桶宽度，默认 0.1")
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_csv)
    overall = summarize_overall(rows)
    bins = build_bins(rows, args.bin_width)

    write_csv(output_dir / "prob_bins.csv", bins)

    summary = {
        "input_csv": str(input_csv),
        "prob_definition": "prob 是正类概率",
        "overall": overall,
        "bins": bins,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("overall")
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("bins")
    print(json.dumps(bins, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
