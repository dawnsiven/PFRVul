import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_PATTERNS = ("results.csv", "reviewer_test.csv", "merged_results.csv")


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 outputs 下各结果 CSV 的分类指标")
    parser.add_argument(
        "--root",
        default="outputs",
        help="递归扫描的根目录，默认 outputs",
    )
    parser.add_argument(
        "--output_dir",
        default="analysis/output_results_summary",
        help="汇总结果输出目录",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=list(DEFAULT_PATTERNS),
        help="需要统计的文件名，默认 results.csv reviewer_test.csv merged_results.csv",
    )
    return parser.parse_args()


def find_result_files(root: Path, patterns: Iterable[str]) -> List[Path]:
    matched: List[Path] = []
    pattern_set = set(patterns)
    for path in sorted(root.rglob("*.csv")):
        if path.name in pattern_set:
            matched.append(path)
    return matched


def compute_metrics(csv_path: Path, scan_root: Path) -> Dict[str, object]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if "Label" not in fieldnames:
            raise ValueError(f"{csv_path} 缺少必要列: ['Label']")

        prediction_column = ""
        for candidate in ("Prediction", "FinalPrediction", "OriginalPrediction", "ReviewerPrediction"):
            if candidate in fieldnames:
                prediction_column = candidate
                break
        if not prediction_column:
            raise ValueError(f"{csv_path} 缺少预测列")

        prob_column = ""
        for candidate in ("prob", "OriginalProb", "ReviewerProb"):
            if candidate in fieldnames:
                prob_column = candidate
                break

        total = 0
        tp = tn = fp = fn = 0
        prob_sum = 0.0
        prob_count = 0

        for row in reader:
            label = int(row["Label"])
            pred = int(row[prediction_column])
            total += 1

            if label == 1 and pred == 1:
                tp += 1
            elif label == 0 and pred == 0:
                tn += 1
            elif label == 0 and pred == 1:
                fp += 1
            elif label == 1 and pred == 0:
                fn += 1

            if prob_column and row[prob_column] != "":
                prob_sum += float(row[prob_column])
                prob_count += 1

    if total == 0:
        raise ValueError(f"{csv_path} 没有有效样本")

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(tp + tn, total)

    relative_path = csv_path.relative_to(scan_root)
    parts = relative_path.parts
    top_level_dir = parts[0] if parts else ""
    experiment_dir = csv_path.parent.name

    return {
        "scan_root": str(scan_root),
        "csv_path": str(csv_path),
        "relative_path": str(relative_path),
        "parent_relative_dir": str(relative_path.parent),
        "top_level_dir": top_level_dir,
        "experiment_dir": experiment_dir,
        "file_name": csv_path.name,
        "prediction_column": prediction_column,
        "prob_column": prob_column,
        "total_samples": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "label_1_count": tp + fn,
        "label_0_count": tn + fp,
        "pred_1_count": tp + fp,
        "pred_0_count": tn + fn,
        "label_1_rate": safe_div(tp + fn, total),
        "pred_1_rate": safe_div(tp + fp, total),
        "avg_prob": safe_div(prob_sum, prob_count) if prob_count else "",
    }


def aggregate_rows(rows: List[Dict[str, object]], group_key: str) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, int]] = {}
    for row in rows:
        key = str(row[group_key])
        bucket = grouped.setdefault(
            key,
            {
                "files": 0,
                "total_samples": 0,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
            },
        )
        bucket["files"] += 1
        bucket["total_samples"] += int(row["total_samples"])
        bucket["tp"] += int(row["tp"])
        bucket["tn"] += int(row["tn"])
        bucket["fp"] += int(row["fp"])
        bucket["fn"] += int(row["fn"])

    summary_rows: List[Dict[str, object]] = []
    for key in sorted(grouped):
        item = grouped[key]
        total = item["total_samples"]
        tp = item["tp"]
        tn = item["tn"]
        fp = item["fp"]
        fn = item["fn"]
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        summary_rows.append(
            {
                group_key: key,
                "files": item["files"],
                "total_samples": total,
                "accuracy": safe_div(tp + tn, total),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "label_1_count": tp + fn,
                "label_0_count": tn + fp,
                "pred_1_count": tp + fp,
                "pred_0_count": tn + fn,
                "label_1_rate": safe_div(tp + fn, total),
                "pred_1_rate": safe_div(tp + fp, total),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scan_root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not scan_root.exists():
        raise SystemExit(f"扫描根目录不存在: {scan_root}")

    result_files = find_result_files(scan_root, args.patterns)
    if not result_files:
        raise SystemExit("未找到可统计的结果 CSV")

    detail_rows = [compute_metrics(path, scan_root) for path in result_files]
    detail_rows.sort(key=lambda row: (str(row["top_level_dir"]), str(row["relative_path"])))

    by_top_level = aggregate_rows(detail_rows, "top_level_dir")
    by_parent_dir = aggregate_rows(detail_rows, "parent_relative_dir")
    by_experiment = aggregate_rows(detail_rows, "experiment_dir")

    write_csv(output_dir / "file_metrics.csv", detail_rows)
    write_csv(output_dir / "top_level_dir_metrics.csv", by_top_level)
    write_csv(output_dir / "parent_dir_metrics.csv", by_parent_dir)
    write_csv(output_dir / "experiment_dir_metrics.csv", by_experiment)

    print(f"Scanned files: {len(detail_rows)}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
