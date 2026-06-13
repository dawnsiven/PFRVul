import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "统计 LLM 误判与原始小模型概率之间的关系。"
            "优先读取 merged_results.csv；如 OriginalProb 缺失，可回表 original results.csv 补齐。"
        )
    )
    parser.add_argument("--merged_csv", required=True, help="路径：merged_results.csv")
    parser.add_argument(
        "--original_results_csv",
        default=None,
        help="可选。原始小模型 results.csv；当 merged_results.csv 里 OriginalProb 为空时，用它按 Index 补齐。",
    )
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument(
        "--bin_width",
        type=float,
        default=0.1,
        help="原始小模型概率分桶宽度，默认 0.1",
    )
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="统计 prob >= threshold 时 LLM 误判率，默认 0.1 到 0.9",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_int(value: object) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def to_float(value: object) -> Optional[float]:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def resolve_prob_column(fieldnames: List[str]) -> Optional[str]:
    for column in ("OriginalProb", "Prob", "prob"):
        if column in fieldnames:
            return column
    return None


def load_original_rows_by_index(original_rows: List[dict]) -> Dict[int, dict]:
    result: Dict[int, dict] = {}
    for row in original_rows:
        index = to_int(row.get("Index"))
        if index is not None:
            result[index] = row
    return result


def load_original_prob_by_index(original_rows: List[dict]) -> Dict[int, float]:
    if not original_rows:
        return {}
    fieldnames = list(original_rows[0].keys())
    prob_column = resolve_prob_column(fieldnames)
    if prob_column is None:
        raise ValueError(f"原始 results.csv 中未找到概率列，当前列: {fieldnames}")

    result: Dict[int, float] = {}
    for row in original_rows:
        index = to_int(row.get("Index"))
        prob = to_float(row.get(prob_column))
        if index is not None and prob is not None:
            result[index] = prob
    return result


def pearson_corr(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n == 0 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def build_bins(rows: List[Dict[str, object]], bin_width: float) -> List[Dict[str, object]]:
    bins: List[Dict[str, object]] = []
    start = 0.0
    while start < 1.0:
        end = min(start + bin_width, 1.0)
        subset = [
            row for row in rows
            if (start <= row["small_model_prob"] < end)
            or (end >= 1.0 and start <= row["small_model_prob"] <= 1.0)
        ]
        count = len(subset)
        if count:
            llm_wrong = sum(row["llm_wrong"] for row in subset)
            small_wrong = sum(row["small_model_wrong"] for row in subset)
            bins.append(
                {
                    "bin_start": round(start, 4),
                    "bin_end": round(end, 4),
                    "count": count,
                    "ratio": round(safe_div(count, len(rows)), 6),
                    "llm_error_rate": round(safe_div(llm_wrong, count), 6),
                    "small_model_error_rate": round(safe_div(small_wrong, count), 6),
                    "avg_small_model_prob": round(
                        safe_div(sum(row["small_model_prob"] for row in subset), count), 6
                    ),
                    "avg_small_model_confidence": round(
                        safe_div(sum(row["small_model_confidence"] for row in subset), count), 6
                    ),
                    "llm_wrong_count": llm_wrong,
                    "small_model_wrong_count": small_wrong,
                }
            )
        start = round(start + bin_width, 10)
    return bins


def build_thresholds(rows: List[Dict[str, object]], thresholds: List[float]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    total = len(rows)
    for threshold in thresholds:
        subset = [row for row in rows if row["small_model_prob"] >= threshold]
        count = len(subset)
        llm_wrong = sum(row["llm_wrong"] for row in subset)
        small_wrong = sum(row["small_model_wrong"] for row in subset)
        result.append(
            {
                "threshold": threshold,
                "count": count,
                "coverage": round(safe_div(count, total), 6),
                "llm_error_rate": round(safe_div(llm_wrong, count), 6),
                "small_model_error_rate": round(safe_div(small_wrong, count), 6),
                "avg_small_model_prob": round(
                    safe_div(sum(row["small_model_prob"] for row in subset), count), 6
                ),
            }
        )
    return result


def validate_original_alignment(
    merged_rows: List[dict],
    original_rows_by_index: Dict[int, dict],
) -> Dict[str, object]:
    checked = 0
    matched = 0
    mismatched_examples: List[Dict[str, object]] = []

    for row in merged_rows:
        index = to_int(row.get("Index"))
        merged_original_pred = to_int(row.get("OriginalPrediction"))
        if index is None or merged_original_pred is None:
            continue
        original_row = original_rows_by_index.get(index)
        if not original_row:
            continue
        original_pred = to_int(original_row.get("Prediction"))
        if original_pred is None:
            continue
        checked += 1
        if original_pred == merged_original_pred:
            matched += 1
        elif len(mismatched_examples) < 10:
            mismatched_examples.append(
                {
                    "Index": index,
                    "merged_original_prediction": merged_original_pred,
                    "results_csv_prediction": original_pred,
                    "results_csv_prob": original_row.get("Prob", original_row.get("prob", "")),
                }
            )

    return {
        "checked_count": checked,
        "matched_count": matched,
        "match_rate": safe_div(matched, checked),
        "mismatched_examples": mismatched_examples,
    }


def build_records(merged_rows: List[dict], original_prob_by_index: Dict[int, float]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for row in merged_rows:
        label = to_int(row.get("Label"))
        llm_pred = to_int(row.get("LLMPrediction"))
        original_pred = to_int(row.get("OriginalPrediction"))
        index = to_int(row.get("Index"))
        merged_prob = to_float(row.get("OriginalProb"))
        prob = merged_prob
        if prob is None and index is not None:
            prob = original_prob_by_index.get(index)

        if label is None or llm_pred is None or original_pred is None or prob is None:
            continue

        records.append(
            {
                "Index": index,
                "Label": label,
                "OriginalPrediction": original_pred,
                "LLMPrediction": llm_pred,
                "small_model_prob": prob,
                "small_model_confidence": max(prob, 1.0 - prob),
                "llm_wrong": 1 if llm_pred != label else 0,
                "small_model_wrong": 1 if original_pred != label else 0,
                "llm_changed_small_model": 1 if llm_pred != original_pred else 0,
                "llm_fixed_small_model_error": 1 if original_pred != label and llm_pred == label else 0,
                "llm_introduced_error": 1 if original_pred == label and llm_pred != label else 0,
            }
        )
    return records


def summarize(records: List[Dict[str, object]], merged_csv: Path, original_results_csv: Optional[Path]) -> Dict[str, object]:
    probs = [row["small_model_prob"] for row in records]
    llm_wrong = [row["llm_wrong"] for row in records]
    small_wrong = [row["small_model_wrong"] for row in records]

    llm_wrong_count = sum(llm_wrong)
    small_wrong_count = sum(small_wrong)

    llm_wrong_probs = [row["small_model_prob"] for row in records if row["llm_wrong"] == 1]
    llm_correct_probs = [row["small_model_prob"] for row in records if row["llm_wrong"] == 0]

    return {
        "merged_csv": str(merged_csv),
        "original_results_csv": None if original_results_csv is None else str(original_results_csv),
        "reviewed_sample_count": len(records),
        "llm_error_count": llm_wrong_count,
        "llm_error_rate": round(safe_div(llm_wrong_count, len(records)), 6),
        "small_model_error_count_within_reviewed_subset": small_wrong_count,
        "small_model_error_rate_within_reviewed_subset": round(safe_div(small_wrong_count, len(records)), 6),
        "llm_changed_prediction_count": sum(row["llm_changed_small_model"] for row in records),
        "llm_fixed_small_model_error_count": sum(row["llm_fixed_small_model_error"] for row in records),
        "llm_introduced_error_count": sum(row["llm_introduced_error"] for row in records),
        "avg_small_model_prob_all": round(safe_div(sum(probs), len(probs)), 6),
        "avg_small_model_prob_when_llm_wrong": round(safe_div(sum(llm_wrong_probs), len(llm_wrong_probs)), 6),
        "avg_small_model_prob_when_llm_correct": round(safe_div(sum(llm_correct_probs), len(llm_correct_probs)), 6),
        "pearson_corr_small_model_prob_vs_llm_wrong": (
            None if pearson_corr(probs, llm_wrong) is None else round(pearson_corr(probs, llm_wrong), 6)
        ),
        "pearson_corr_small_model_prob_vs_small_model_wrong": (
            None if pearson_corr(probs, small_wrong) is None else round(pearson_corr(probs, small_wrong), 6)
        ),
    }


def main() -> None:
    args = parse_args()

    merged_csv = Path(args.merged_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    merged_rows = load_csv(merged_csv)
    original_results_csv = Path(args.original_results_csv).resolve() if args.original_results_csv else None
    original_prob_by_index: Dict[int, float] = {}
    original_rows_by_index: Dict[int, dict] = {}
    alignment_summary: Optional[Dict[str, object]] = None
    if original_results_csv is not None:
        original_rows = load_csv(original_results_csv)
        original_prob_by_index = load_original_prob_by_index(original_rows)
        original_rows_by_index = load_original_rows_by_index(original_rows)
        alignment_summary = validate_original_alignment(merged_rows, original_rows_by_index)
        if alignment_summary["checked_count"] > 0 and alignment_summary["match_rate"] < 0.95:
            raise ValueError(
                "提供的 --original_results_csv 与 merged_results.csv 很可能不是同一次实验。"
                f" prediction 对齐率只有 {alignment_summary['match_rate']:.4f}。"
                " 请换成真正对应的小模型 results.csv。"
            )

    records = build_records(merged_rows, original_prob_by_index)
    if not records:
        raise ValueError(
            "没有可用于统计的样本。请检查 merged_results.csv 中是否有有效 LLMPrediction，"
            "以及是否需要通过 --original_results_csv 补齐小模型概率。"
        )

    summary = summarize(records, merged_csv, original_results_csv)
    bins = build_bins(records, args.bin_width)
    thresholds = build_thresholds(records, args.thresholds)

    write_csv(output_dir / "llm_vs_small_model_prob_records.csv", records)
    write_csv(output_dir / "llm_vs_small_model_prob_bins.csv", bins)
    write_csv(output_dir / "llm_vs_small_model_prob_thresholds.csv", thresholds)

    with (output_dir / "llm_vs_small_model_prob_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary,
                "alignment_summary": alignment_summary,
                "bins": bins,
                "thresholds": thresholds,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("thresholds")
    print(json.dumps(thresholds, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
