import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DEFAULT_INSTRUCTION = "Detect whether the following code contains vulnerabilities."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct fuller method-level samples from chunk-labeled test data and "
            "aggregate chunk-level model outputs into method-level evaluation files."
        )
    )
    parser.add_argument(
        "--data_json",
        required=True,
        help="Path to chunk-level test JSON/JSONL, e.g. data/chunk_lable/uploads/..._test.json",
    )
    parser.add_argument(
        "--results_csv",
        required=True,
        help="Path to chunk-level model results.csv aligned by Index.",
    )
    parser.add_argument(
        "--pairs_jsonl",
        default=None,
        help=(
            "Optional pairs JSONL with full before/after code, e.g. "
            "data/chunk_lable/pairs/cvefixes_local_fix_pairs_test.jsonl. "
            "When provided, the reconstructed sample input prefers pairs long code "
            "over chunk concatenation."
        ),
    )
    parser.add_argument(
        "--output_root",
        default="LLM_TEST/intermediate",
        help="Root directory for reconstructed outputs.",
    )
    parser.add_argument(
        "--output_subdir",
        default=None,
        help="Output subdirectory under output_root. Defaults to reconstructed_<data_json_stem>.",
    )
    parser.add_argument(
        "--positive_prediction_value",
        type=int,
        default=1,
        help="Which aggregated prediction value counts as a positive sample for LLM review.",
    )
    return parser.parse_args()


def load_json_like(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows: List[dict] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
        return rows
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError(f"Unsupported JSON root type in {path}: {type(payload).__name__}")


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


def get_prob(row: dict) -> Optional[float]:
    return to_float(row.get("Prob", row.get("prob")))


def group_key(row: dict) -> Tuple[str, str, str]:
    return (
        str(row.get("pair_id", "")),
        str(row.get("side", "")),
        str(row.get("method_change_id", "")),
    )


def build_pairs_code_map(path: Optional[Path]) -> Dict[Tuple[str, str, str], str]:
    if path is None:
        return {}

    mapping: Dict[Tuple[str, str, str], str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            pair_id = str(record.get("pair_id", ""))
            for side in ("before", "after"):
                node = record.get(side) or {}
                method_change_id = str(node.get("method_change_id", ""))
                code = str(node.get("code", ""))
                if pair_id and method_change_id and code.strip():
                    mapping[(pair_id, side, method_change_id)] = code
    return mapping


def join_chunks(rows: List[dict]) -> str:
    ordered = sorted(
        rows,
        key=lambda item: (
            to_int(item.get("chunk_start_line"), default=10**9),
            to_int(item.get("chunk_id"), default=10**9),
            to_int(item.get("index"), default=10**9),
        ),
    )
    return "\n".join(str(row.get("input", "")).rstrip("\n") for row in ordered if str(row.get("input", "")).strip())


def aggregate_group(
    group_rows: List[dict],
    results_by_index: Dict[int, dict],
    reconstructed_index: int,
    pairs_code_map: Dict[Tuple[str, str, str], str],
) -> Tuple[dict, dict]:
    ordered = sorted(
        group_rows,
        key=lambda item: (
            to_int(item.get("chunk_start_line"), default=10**9),
            to_int(item.get("chunk_id"), default=10**9),
            to_int(item.get("index"), default=10**9),
        ),
    )
    sample_indexes = [to_int(row.get("index"), default=None) for row in ordered]
    chunk_results = [results_by_index[idx] for idx in sample_indexes if idx is not None and idx in results_by_index]

    label_values = [to_int(row.get("output", row.get("label")), default=0) or 0 for row in ordered]
    pred_values = [to_int(row.get("Prediction"), default=0) or 0 for row in chunk_results]
    prob_values = [prob for prob in (get_prob(row) for row in chunk_results) if prob is not None]

    aggregated_label = max(label_values) if label_values else 0
    aggregated_prediction = max(pred_values) if pred_values else 0
    aggregated_probability = max(prob_values) if prob_values else None

    first = ordered[0]
    lookup_key = (
        str(first.get("pair_id", "")),
        str(first.get("side", "")),
        str(first.get("method_change_id", "")),
    )
    pairs_full_code = pairs_code_map.get(lookup_key)
    raw_sample = {
        "pair_id": first.get("pair_id"),
        "pair_key": first.get("pair_key"),
        "cve_id": first.get("cve_id"),
        "cwe_id": first.get("cwe_id"),
        "commit_hash": first.get("commit_hash"),
        "file_change_id": first.get("file_change_id"),
        "filename": first.get("filename"),
        "programming_language": first.get("programming_language"),
        "change_type": first.get("change_type"),
        "changed_lines": first.get("changed_lines"),
        "side": first.get("side"),
        "method_change_id": first.get("method_change_id"),
        "method_name": first.get("method_name"),
        "method_signature": first.get("method_signature"),
        "chunk_count": len(ordered),
        "chunk_indexes": sample_indexes,
        "code_source": "pairs_full_code" if pairs_full_code else "stitched_chunks",
        "chunk_ranges": [
            {
                "chunk_id": row.get("chunk_id"),
                "chunk_start_line": row.get("chunk_start_line"),
                "chunk_end_line": row.get("chunk_end_line"),
            }
            for row in ordered
        ],
    }
    reconstructed_input = pairs_full_code or join_chunks(ordered)

    prepared_sample = {
        "dataset_id": "",
        "index": reconstructed_index,
        "sample_index": sample_indexes,
        "matched_by": "reconstructed_method_group",
        "cwe": first.get("cwe_id", "null"),
        "ground_truth": aggregated_label,
        "original_prediction": aggregated_prediction,
        "original_probability": aggregated_probability,
        "instruction": first.get("instruction", DEFAULT_INSTRUCTION),
        "input": reconstructed_input,
        "output": str(aggregated_label),
        "raw_sample": raw_sample,
    }

    aggregated_result = {
        "Index": reconstructed_index,
        "Label": aggregated_label,
        "Prediction": aggregated_prediction,
        "Prob": "" if aggregated_probability is None else aggregated_probability,
        "CWE": first.get("cwe_id", "null"),
        "PairId": first.get("pair_id", ""),
        "Side": first.get("side", ""),
        "MethodChangeId": first.get("method_change_id", ""),
        "MethodName": first.get("method_name", ""),
        "MethodSignature": first.get("method_signature", ""),
        "ChunkCount": len(ordered),
        "ChunkIndexes": json.dumps(sample_indexes, ensure_ascii=False),
    }
    return prepared_sample, aggregated_result


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_dataset_id(data_json: Path) -> str:
    stem = data_json.stem
    if stem.endswith("_test"):
        stem = stem[:-5]
    return f"reconstructed_{stem}"


def main() -> None:
    args = parse_args()
    data_json = Path(args.data_json).resolve()
    results_csv = Path(args.results_csv).resolve()
    pairs_jsonl = Path(args.pairs_jsonl).resolve() if args.pairs_jsonl else None
    if not data_json.exists():
        raise FileNotFoundError(f"Data JSON not found: {data_json}")
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")
    if pairs_jsonl is not None and not pairs_jsonl.exists():
        raise FileNotFoundError(f"pairs_jsonl not found: {pairs_jsonl}")

    output_root = Path(args.output_root).resolve()
    output_subdir = args.output_subdir or f"reconstructed_{data_json.stem}"
    output_dir = output_root / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_rows = load_json_like(data_json)
    results_by_index = load_results(results_csv)
    pairs_code_map = build_pairs_code_map(pairs_jsonl)

    grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for row in data_rows:
        grouped[group_key(row)].append(row)

    prepared_rows: List[dict] = []
    aggregated_results: List[dict] = []
    missing_result_chunks = 0
    pairs_code_hits = 0

    for reconstructed_index, key in enumerate(sorted(grouped.keys())):
        group_rows = grouped[key]
        sample, aggregated_result = aggregate_group(
            group_rows,
            results_by_index,
            reconstructed_index,
            pairs_code_map,
        )
        matched_chunk_results = sum(
            1
            for idx in sample["sample_index"]
            if idx is not None and idx in results_by_index
        )
        missing_result_chunks += max(0, len(sample["sample_index"]) - matched_chunk_results)
        if sample["raw_sample"].get("code_source") == "pairs_full_code":
            pairs_code_hits += 1
        sample["dataset_id"] = infer_dataset_id(data_json)
        prepared_rows.append(sample)
        aggregated_results.append(aggregated_result)

    positive_rows = [
        row for row in prepared_rows
        if to_int(row.get("original_prediction"), default=0) == args.positive_prediction_value
    ]

    write_jsonl(output_dir / "full_test_samples.jsonl", prepared_rows)
    write_jsonl(output_dir / "positive_samples.jsonl", positive_rows)
    write_csv(
        output_dir / "grouped_results.csv",
        aggregated_results,
        [
            "Index",
            "Label",
            "Prediction",
            "Prob",
            "CWE",
            "PairId",
            "Side",
            "MethodChangeId",
            "MethodName",
            "MethodSignature",
            "ChunkCount",
            "ChunkIndexes",
        ],
    )
    write_json(
        output_dir / "summary.json",
        {
            "dataset_id": infer_dataset_id(data_json),
            "source_data_json": str(data_json),
            "source_results_csv": str(results_csv),
            "total_chunk_rows": len(data_rows),
            "total_reconstructed_samples": len(prepared_rows),
            "positive_reconstructed_samples": len(positive_rows),
            "missing_result_chunks": missing_result_chunks,
            "pairs_jsonl": str(pairs_jsonl) if pairs_jsonl else None,
            "pairs_full_code_hits": pairs_code_hits,
            "positive_prediction_value": args.positive_prediction_value,
            "output_files": {
                "grouped_results_csv": str(output_dir / "grouped_results.csv"),
                "positive_samples_jsonl": str(output_dir / "positive_samples.jsonl"),
                "full_test_samples_jsonl": str(output_dir / "full_test_samples.jsonl"),
            },
        },
    )

    print(f"output_dir={output_dir}")
    print(f"grouped_results_csv={output_dir / 'grouped_results.csv'}")
    print(f"positive_samples_jsonl={output_dir / 'positive_samples.jsonl'}")
    print(f"full_test_samples_jsonl={output_dir / 'full_test_samples.jsonl'}")
    print(f"total_reconstructed_samples={len(prepared_rows)}")
    print(f"positive_reconstructed_samples={len(positive_rows)}")


if __name__ == "__main__":
    main()
