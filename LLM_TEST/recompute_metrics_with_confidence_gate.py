import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from config_utils import get_section, load_env_file, load_yaml_config, resolve_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep the small-model decision when its confidence is above a threshold; "
            "otherwise defer to the LLM prediction when available."
        )
    )
    parser.add_argument("--config", default="LLM_TEST/exp.yaml", help="Path to YAML config.")
    parser.add_argument("--env_file", default="LLM_TEST/.env", help="Path to environment config file.")
    parser.add_argument("--results_csv", default=None, help="Path to the original results.csv.")
    parser.add_argument(
        "--llm_predictions_csv",
        default=None,
        help="Path to LLM_TEST/output/<dataset_id>/llm_predictions.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for merged outputs. Defaults to the parent directory of --llm_predictions_csv.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.9,
        help="Keep the small-model decision when confidence >= threshold.",
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


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def judgment_confidence(prediction: int, positive_prob: float) -> float:
    if prediction == 1:
        return positive_prob
    return 1.0 - positive_prob


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    config = load_yaml_config(args.config)
    metrics_cfg = get_section(config, "metrics")

    results_csv = Path(resolve_value(args.results_csv, metrics_cfg, "results_csv")).resolve()
    llm_predictions_csv = Path(
        resolve_value(args.llm_predictions_csv, metrics_cfg, "llm_predictions_csv")
    ).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else Path(resolve_value(None, metrics_cfg, "output_dir", llm_predictions_csv.parent)).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    original_rows = load_csv(results_csv)
    llm_rows = load_csv(llm_predictions_csv)
    llm_by_index = {to_int(row.get("Index")): row for row in llm_rows}

    merged_rows: List[dict] = []
    labels: List[int] = []
    original_preds: List[int] = []
    gated_preds: List[int] = []

    kept_small_model_count = 0
    deferred_to_llm_count = 0
    fallback_to_small_model_count = 0
    llm_used_count = 0
    llm_changed_decision_count = 0

    confident_positive_kept = 0
    confident_negative_kept = 0
    low_conf_positive_sent_to_llm = 0
    low_conf_negative_sent_to_llm = 0

    for row in original_rows:
        index = to_int(row.get("Index"))
        label = to_int(row.get("Label"))
        original_pred = to_int(row.get("Prediction"))
        prob = to_float(row.get("Prob"))
        confidence = judgment_confidence(original_pred, prob)
        llm_row = llm_by_index.get(index)

        final_pred = original_pred
        decision_source = "small_model_high_confidence"

        if confidence >= args.confidence_threshold:
            kept_small_model_count += 1
            if original_pred == 1:
                confident_positive_kept += 1
            else:
                confident_negative_kept += 1
        else:
            deferred_to_llm_count += 1
            if original_pred == 1:
                low_conf_positive_sent_to_llm += 1
            else:
                low_conf_negative_sent_to_llm += 1

            if llm_row and str(llm_row.get("LLMPrediction", "")).strip() in {"0", "1"}:
                final_pred = to_int(llm_row.get("LLMPrediction"))
                decision_source = "llm_low_confidence_override"
                llm_used_count += 1
                if final_pred != original_pred:
                    llm_changed_decision_count += 1
            else:
                decision_source = "small_model_low_confidence_fallback"
                fallback_to_small_model_count += 1

        labels.append(label)
        original_preds.append(original_pred)
        gated_preds.append(final_pred)

        merged_rows.append(
            {
                "Index": index,
                "CWE": row.get("CWE", "null"),
                "Label": label,
                "OriginalPrediction": original_pred,
                "FinalPrediction": final_pred,
                "OriginalProb": row.get("Prob", ""),
                "SmallModelConfidence": f"{confidence:.6f}",
                "DecisionSource": decision_source,
                "LLMPrediction": "" if not llm_row else llm_row.get("LLMPrediction", ""),
                "LLMModel": "" if not llm_row else llm_row.get("LLMModel", ""),
                "LLMParseStatus": "" if not llm_row else llm_row.get("ParseStatus", ""),
            }
        )

    original_metrics = compute_metrics(labels, original_preds)
    confidence_gated_metrics = compute_metrics(labels, gated_preds)

    summary = {
        "confidence_threshold": args.confidence_threshold,
        "total_samples": len(labels),
        "kept_small_model_count": kept_small_model_count,
        "deferred_to_llm_count": deferred_to_llm_count,
        "llm_used_count": llm_used_count,
        "fallback_to_small_model_count": fallback_to_small_model_count,
        "llm_changed_decision_count": llm_changed_decision_count,
        "kept_small_model_rate": safe_rate(kept_small_model_count, len(labels)),
        "deferred_to_llm_rate": safe_rate(deferred_to_llm_count, len(labels)),
        "llm_used_rate": safe_rate(llm_used_count, len(labels)),
        "llm_changed_decision_rate": safe_rate(llm_changed_decision_count, len(labels)),
        "confident_positive_kept": confident_positive_kept,
        "confident_negative_kept": confident_negative_kept,
        "low_conf_positive_sent_to_llm": low_conf_positive_sent_to_llm,
        "low_conf_negative_sent_to_llm": low_conf_negative_sent_to_llm,
    }

    merged_csv_path = output_dir / "merged_results_confidence_gate.csv"
    with merged_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Index",
                "CWE",
                "Label",
                "OriginalPrediction",
                "FinalPrediction",
                "OriginalProb",
                "SmallModelConfidence",
                "DecisionSource",
                "LLMPrediction",
                "LLMModel",
                "LLMParseStatus",
            ],
        )
        writer.writeheader()
        writer.writerows(merged_rows)

    metrics_path = output_dir / "metrics_confidence_gate.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "results_csv": str(results_csv),
                "llm_predictions_csv": str(llm_predictions_csv),
                "merged_results_csv": str(merged_csv_path),
                "strategy": (
                    "keep the small-model decision when confidence >= threshold; "
                    "otherwise use the LLM prediction when available; if LLM is missing, fallback to the small model"
                ),
                "confidence_definition": (
                    "if Prediction == 1 use Prob; if Prediction == 0 use 1 - Prob"
                ),
                "original_metrics": original_metrics,
                "confidence_gated_metrics": confidence_gated_metrics,
                "summary": summary,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print("original_metrics")
    print(json.dumps(original_metrics, ensure_ascii=False, indent=2))
    print("confidence_gated_metrics")
    print(json.dumps(confidence_gated_metrics, ensure_ascii=False, indent=2))
    print("summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"merged_results={merged_csv_path}")
    print(f"metrics_json={metrics_path}")


if __name__ == "__main__":
    main()
