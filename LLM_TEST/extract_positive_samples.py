import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from config_utils import get_section, load_env_file, load_yaml_config, resolve_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract positive predictions from results.csv and map them to the source test JSON."
    )
    parser.add_argument("--config", default="LLM_TEST/exp.yaml", help="Path to YAML config.")
    parser.add_argument("--env_file", default="LLM_TEST/.env", help="Path to environment config file.")
    parser.add_argument("--results_csv", default=None, help="Path to the original results.csv file.")
    parser.add_argument(
        "--data_json",
        default=None,
        help="Path to the source *_test.json file. If omitted, the script will try to infer it.",
    )
    parser.add_argument(
        "--data_root",
        default="data",
        help="Root directory used to infer the source *_test.json file when --data_json is omitted.",
    )
    parser.add_argument(
        "--output_root",
        default="LLM_TEST/intermediate",
        help="Root directory for intermediate outputs.",
    )
    parser.add_argument(
        "--output_subdir",
        default=None,
        help="Custom output subdirectory name under output_root. Defaults to the inferred dataset_id.",
    )
    parser.add_argument(
        "--prediction_value",
        type=int,
        default=1,
        help="Only keep rows whose Prediction equals this value. Defaults to 1.",
    )
    parser.add_argument(
        "--min_prob",
        type=float,
        default=None,
        help="Optional inclusive lower bound for Prob filtering.",
    )
    parser.add_argument(
        "--max_prob",
        type=float,
        default=None,
        help="Optional inclusive upper bound for Prob filtering.",
    )
    return parser.parse_args()


def infer_dataset_id(results_csv: Path) -> str:
    return results_csv.parent.name


def infer_data_json(results_csv: Path, data_root: Path) -> Path:
    dataset_id = infer_dataset_id(results_csv)
    candidates = sorted(data_root.rglob(f"{dataset_id}_test.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not infer data json for dataset_id={dataset_id} under {data_root}"
        )
    if len(candidates) > 1:
        raise FileExistsError(
            f"Found multiple candidate test json files for dataset_id={dataset_id}: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def resolve_data_json_path(
    cli_data_json: Optional[str],
    config_data_json: Optional[str],
    results_csv: Path,
    data_root: Path,
) -> Path:
    if cli_data_json:
        return Path(cli_data_json).resolve()

    try:
        return infer_data_json(results_csv, data_root)
    except (FileNotFoundError, FileExistsError):
        if config_data_json:
            return Path(config_data_json).resolve()
        raise


def load_results(results_csv: Path) -> List[dict]:
    with results_csv.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(data_json: Path) -> List[dict]:
    with data_json.open("r", encoding="utf-8") as handle:
        text = handle.read().strip()

    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows: List[dict] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Unsupported JSON format in {data_json} at line {line_no}. "
                    "Expected a JSON array/object or JSONL with one JSON object per line."
                ) from exc
        return rows

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON root type in {data_json}: {type(payload).__name__}")


def build_index_map(records: List[dict]) -> Dict[int, dict]:
    mapping: Dict[int, dict] = {}
    for record in records:
        record_index = record.get("index")
        if record_index is None:
            continue
        mapping[int(record_index)] = record
    return mapping


def to_int(value: object, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def to_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def get_probability(row: dict) -> Optional[float]:
    # Some result CSVs use "Prob" while others use lowercase "prob".
    return to_float(row.get("Prob", row.get("prob")))


def safe_write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def safe_write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    config = load_yaml_config(args.config)
    common_cfg = get_section(config, "common")
    extract_cfg = get_section(config, "extract")

    results_csv = Path(resolve_value(args.results_csv, extract_cfg, "results_csv")).resolve()
    data_root = Path(resolve_value(args.data_root, common_cfg, "data_root", "data")).resolve()
    output_root = resolve_value(args.output_root, common_cfg, "intermediate_root", "LLM_TEST/intermediate")
    config_data_json = extract_cfg.get("data_json")
    data_json = resolve_data_json_path(
        cli_data_json=args.data_json,
        config_data_json=config_data_json,
        results_csv=results_csv,
        data_root=data_root,
    )

    dataset_id = infer_dataset_id(results_csv)
    output_subdir = args.output_subdir or dataset_id
    output_dir = Path(output_root).resolve() / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    result_rows = load_results(results_csv)
    data_rows = load_json(data_json)
    data_by_index = build_index_map(data_rows)

    positive_rows = [row for row in result_rows if to_int(row.get("Prediction")) == args.prediction_value]
    filtered_rows: List[dict] = []
    for row in positive_rows:
        prob = get_probability(row)
        if args.min_prob is not None and (prob is None or prob < args.min_prob):
            continue
        if args.max_prob is not None and (prob is None or prob > args.max_prob):
            continue
        filtered_rows.append(row)

    matched_samples: List[dict] = []
    unmatched_indices: List[int] = []
    positive_indices: List[int] = []

    for row in filtered_rows:
        sample_index = to_int(row.get("Index"))
        positive_indices.append(sample_index)
        sample = data_by_index.get(sample_index)
        matched_by = "record_index"
        if sample is None and 0 <= sample_index < len(data_rows):
            # Some results.csv files use row position rather than the original sample["index"].
            sample = data_rows[sample_index]
            matched_by = "row_position"
        if sample is None:
            unmatched_indices.append(sample_index)
            continue

        matched_samples.append(
            {
                "dataset_id": dataset_id,
                "index": sample_index,
                "sample_index": sample.get("index"),
                "matched_by": matched_by,
                "cwe": row.get("CWE", sample.get("cwe", "null")),
                "ground_truth": to_int(row.get("Label"), default=to_int(sample.get("output"), default=0)),
                "original_prediction": to_int(row.get("Prediction")),
                "original_probability": get_probability(row),
                "instruction": sample.get("instruction", ""),
                "input": sample.get("input", ""),
                "output": sample.get("output", ""),
                "raw_sample": sample,
            }
        )

    safe_write_json(
        output_dir / "summary.json",
        {
            "dataset_id": dataset_id,
            "output_subdir": output_subdir,
            "results_csv": str(results_csv),
            "data_json": str(data_json),
            "total_results": len(result_rows),
            "positive_predictions": len(positive_rows),
            "filtered_predictions": len(filtered_rows),
            "matched_positive_samples": len(matched_samples),
            "unmatched_positive_indices": len(unmatched_indices),
            "prediction_value": args.prediction_value,
            "min_prob": args.min_prob,
            "max_prob": args.max_prob,
        },
    )
    safe_write_json(output_dir / "positive_indices.json", positive_indices)
    safe_write_json(output_dir / "unmatched_indices.json", unmatched_indices)
    safe_write_jsonl(output_dir / "positive_samples.jsonl", matched_samples)

    print(f"dataset_id={dataset_id}")
    print(f"results_csv={results_csv}")
    print(f"data_json={data_json}")
    print(f"positive_predictions={len(positive_rows)}")
    print(f"filtered_predictions={len(filtered_rows)}")
    print(f"matched_positive_samples={len(matched_samples)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
