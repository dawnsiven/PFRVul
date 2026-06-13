import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from config_utils import get_section, load_env_file, load_yaml_config, resolve_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a full test set JSONL file for direct LLM vulnerability detection."
    )
    parser.add_argument("--config", default="LLM_TEST/exp.yaml", help="Path to YAML config.")
    parser.add_argument("--env_file", default="LLM_TEST/.env", help="Path to environment config file.")
    parser.add_argument(
        "--data_json",
        required=True,
        help="Path to the source *_test.json file in Alpaca-style format.",
    )
    parser.add_argument(
        "--results_csv",
        default=None,
        help="Optional results.csv to attach original model predictions and probabilities.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Root directory for prepared JSONL outputs. Defaults to common.intermediate_root.",
    )
    parser.add_argument(
        "--output_subdir",
        default=None,
        help="Custom output subdirectory name under output_root. Defaults to fulltest_<data_json_stem>.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only keep the first N samples.",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {path}, got {type(payload).__name__}.")
    return payload


def load_results(path: Path) -> Dict[int, dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping: Dict[int, dict] = {}
    for row in rows:
        index = to_int(row.get("Index"), default=None)
        if index is None:
            continue
        mapping[index] = row
    return mapping


def to_int(value: object, default: Optional[int] = 0) -> Optional[int]:
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
    return to_float(row.get("Prob", row.get("prob")))


def safe_write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def safe_write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def infer_dataset_id(data_json: Path) -> str:
    stem = data_json.stem
    if stem.endswith("_test"):
        return stem[:-5]
    return stem


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    config = load_yaml_config(args.config)
    common_cfg = get_section(config, "common")

    data_json = Path(args.data_json).resolve()
    if not data_json.exists():
        raise FileNotFoundError(f"Data JSON not found: {data_json}")

    results_csv = Path(args.results_csv).resolve() if args.results_csv else None
    if results_csv and not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    output_root_value = resolve_value(
        args.output_root,
        common_cfg,
        "intermediate_root",
        "LLM_TEST/intermediate",
    )
    output_root = Path(output_root_value).resolve()
    output_subdir = args.output_subdir or f"fulltest_{data_json.stem}"
    output_dir = output_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = infer_dataset_id(data_json)
    data_rows = load_json(data_json)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        data_rows = data_rows[: args.limit]

    results_by_index = load_results(results_csv) if results_csv else {}

    prepared_rows: List[dict] = []
    matched_results = 0
    missing_index_count = 0

    for row_position, sample in enumerate(data_rows):
        sample_index = to_int(sample.get("index"), default=None)
        if sample_index is None:
            sample_index = row_position
            missing_index_count += 1

        result_row = results_by_index.get(sample_index)
        if result_row:
            matched_results += 1

        prepared_rows.append(
            {
                "dataset_id": dataset_id,
                "index": sample_index,
                "sample_index": sample.get("index"),
                "matched_by": "sample_index" if sample.get("index") is not None else "row_position",
                "cwe": result_row.get("CWE", "null") if result_row else sample.get("cwe", "null"),
                "ground_truth": to_int(sample.get("output"), default=0),
                "original_prediction": to_int(result_row.get("Prediction"), default=-1) if result_row else -1,
                "original_probability": get_probability(result_row) if result_row else None,
                "instruction": sample.get("instruction", ""),
                "input": sample.get("input", ""),
                "output": sample.get("output", ""),
                "raw_sample": sample,
            }
        )

    safe_write_jsonl(output_dir / "full_test_samples.jsonl", prepared_rows)
    safe_write_json(
        output_dir / "summary.json",
        {
            "dataset_id": dataset_id,
            "data_json": str(data_json),
            "results_csv": str(results_csv) if results_csv else None,
            "output_subdir": output_subdir,
            "total_samples": len(prepared_rows),
            "positive_labels": sum(row["ground_truth"] == 1 for row in prepared_rows),
            "negative_labels": sum(row["ground_truth"] == 0 for row in prepared_rows),
            "matched_results_rows": matched_results,
            "missing_index_count": missing_index_count,
        },
    )

    print(f"dataset_id={dataset_id}")
    print(f"data_json={data_json}")
    print(f"results_csv={results_csv if results_csv else 'None'}")
    print(f"total_samples={len(prepared_rows)}")
    print(f"output_dir={output_dir}")
    print(f"input_json={output_dir / 'full_test_samples.jsonl'}")


if __name__ == "__main__":
    main()
