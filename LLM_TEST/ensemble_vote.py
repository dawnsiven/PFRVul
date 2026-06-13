import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble multiple llm_predictions.csv files with configurable voting."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input specs in the form name=path/to/llm_predictions.csv",
    )
    parser.add_argument(
        "--strategy",
        choices=["any", "majority", "threshold", "weighted"],
        default="majority",
        help="Voting strategy.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold used by threshold/weighted strategy. "
        "For threshold, it is the minimum number of positive votes. "
        "For weighted, it is the minimum weighted score.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        default=[],
        help="Optional weights for weighted voting, in the form name=weight.",
    )
    parser.add_argument(
        "--intersection_only",
        action="store_true",
        help="Use only samples that have valid predictions in every input model.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for ensemble_predictions.csv and ensemble_metrics.json.",
    )
    parser.add_argument(
        "--output_prefix",
        default="ensemble",
        help="Prefix for generated files.",
    )
    return parser.parse_args()


def parse_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_metrics(labels: Sequence[int], preds: Sequence[int]) -> Dict[str, float]:
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


def compute_llm_only_metrics(labels: Sequence[int], preds: Sequence[int], original_preds: Sequence[int]) -> Dict[str, float]:
    metrics = compute_metrics(labels, preds)
    metrics.update(
        {
            "scope": "llm_ensemble_only",
            "description": "Metrics computed only on samples used by the ensemble vote.",
            "positive_labels": sum(label == 1 for label in labels),
            "negative_labels": sum(label == 0 for label in labels),
            "positive_predictions": sum(pred == 1 for pred in preds),
            "negative_predictions": sum(pred == 0 for pred in preds),
            "changed_vs_original": sum(pred != original for pred, original in zip(preds, original_preds)),
            "kept_vs_original": sum(pred == original for pred, original in zip(preds, original_preds)),
            "actual_positive_kept_by_llm": sum(label == 1 and pred == 1 for label, pred in zip(labels, preds)),
            "actual_positive_rejected_by_llm": sum(label == 1 and pred == 0 for label, pred in zip(labels, preds)),
            "actual_negative_predicted_positive": sum(label == 0 and pred == 1 for label, pred in zip(labels, preds)),
            "actual_negative_predicted_negative": sum(label == 0 and pred == 0 for label, pred in zip(labels, preds)),
        }
    )
    return metrics


def parse_input_specs(input_specs: Sequence[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    for spec in input_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --inputs item: {spec}. Expected name=path")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip()).resolve()
        if not name:
            raise ValueError(f"Invalid model name in --inputs item: {spec}")
        parsed.append((name, path))
    return parsed


def parse_weights(weight_specs: Sequence[str], model_names: Sequence[str]) -> Dict[str, float]:
    weights = {name: 1.0 for name in model_names}
    for spec in weight_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --weights item: {spec}. Expected name=weight")
        name, raw_weight = spec.split("=", 1)
        name = name.strip()
        if name not in weights:
            raise ValueError(f"Weight provided for unknown model: {name}")
        weights[name] = float(raw_weight.strip())
    return weights


def choose_prediction(
    strategy: str,
    votes: Dict[str, int],
    threshold: Optional[float],
    weights: Dict[str, float],
) -> Tuple[int, float]:
    positive_votes = sum(votes.values())
    total_votes = len(votes)
    weighted_score = sum(weights[name] * pred for name, pred in votes.items())

    if strategy == "any":
        return (1 if positive_votes >= 1 else 0), float(positive_votes)
    if strategy == "majority":
        return (1 if positive_votes >= (total_votes // 2 + 1) else 0), float(positive_votes)
    if strategy == "threshold":
        if threshold is None:
            raise ValueError("--threshold is required when --strategy threshold")
        return (1 if positive_votes >= threshold else 0), float(positive_votes)
    if strategy == "weighted":
        if threshold is None:
            threshold = sum(weights.values()) / 2.0
        return (1 if weighted_score >= threshold else 0), weighted_score
    raise ValueError(f"Unsupported strategy: {strategy}")


def load_prediction_tables(input_specs: Sequence[Tuple[str, Path]]) -> Dict[str, Dict[int, dict]]:
    tables: Dict[str, Dict[int, dict]] = {}
    for name, path in input_specs:
        rows = load_csv(path)
        table: Dict[int, dict] = {}
        for row in rows:
            index = parse_int(row.get("Index"))
            label = parse_int(row.get("Label"))
            pred = parse_int(row.get("LLMPrediction"))
            original_pred = parse_int(row.get("OriginalPrediction"))
            if index is None or label is None or pred is None:
                continue
            table[index] = {
                "Index": index,
                "Label": label,
                "OriginalPrediction": 0 if original_pred is None else original_pred,
                "LLMPrediction": pred,
                "LLMModel": row.get("LLMModel", ""),
                "ParseStatus": row.get("ParseStatus", ""),
                "SourceFile": str(path),
            }
        tables[name] = table
    return tables


def build_ensemble_rows(
    tables: Dict[str, Dict[int, dict]],
    strategy: str,
    threshold: Optional[float],
    weights: Dict[str, float],
    intersection_only: bool,
) -> List[dict]:
    model_names = list(tables.keys())
    if intersection_only:
        indices = sorted(set.intersection(*(set(table.keys()) for table in tables.values())))
    else:
        indices = sorted(set.union(*(set(table.keys()) for table in tables.values())))

    rows: List[dict] = []
    for index in indices:
        present = {name: tables[name][index] for name in model_names if index in tables[name]}
        if not present:
            continue

        label = next(item["Label"] for item in present.values())
        original_pred = next(item["OriginalPrediction"] for item in present.values())
        if any(item["Label"] != label for item in present.values()):
            raise ValueError(f"Inconsistent labels for Index={index}")

        if intersection_only and len(present) != len(model_names):
            continue

        votes = {name: item["LLMPrediction"] for name, item in present.items()}
        final_pred, score = choose_prediction(strategy, votes, threshold, weights)
        row = {
            "Index": index,
            "Label": label,
            "OriginalPrediction": original_pred,
            "EnsemblePrediction": final_pred,
            "Strategy": strategy,
            "Score": score,
            "PresentModels": ",".join(sorted(present.keys())),
            "PositiveVotes": sum(votes.values()),
            "TotalVotes": len(votes),
        }
        for name in model_names:
            row[f"Vote_{name}"] = "" if name not in present else present[name]["LLMPrediction"]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_specs = parse_input_specs(args.inputs)
    model_names = [name for name, _ in input_specs]
    weights = parse_weights(args.weights, model_names)
    tables = load_prediction_tables(input_specs)
    rows = build_ensemble_rows(
        tables=tables,
        strategy=args.strategy,
        threshold=args.threshold,
        weights=weights,
        intersection_only=args.intersection_only,
    )
    if not rows:
        raise ValueError("No valid rows available for ensemble voting.")

    labels = [int(row["Label"]) for row in rows]
    preds = [int(row["EnsemblePrediction"]) for row in rows]
    original_preds = [int(row["OriginalPrediction"]) for row in rows]
    metrics = compute_llm_only_metrics(labels, preds, original_preds)
    metrics.update(
        {
            "strategy": args.strategy,
            "threshold": args.threshold,
            "weights": weights,
            "intersection_only": args.intersection_only,
            "models": {name: str(path) for name, path in input_specs},
        }
    )

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = input_specs[0][1].resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / f"{args.output_prefix}_predictions.csv"
    metrics_path = output_dir / f"{args.output_prefix}_metrics.json"
    write_csv(predictions_path, rows)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(f"Wrote predictions to: {predictions_path}")
    print(f"Wrote metrics to: {metrics_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
