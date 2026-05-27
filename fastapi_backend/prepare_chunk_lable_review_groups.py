from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
LLM_TEST_INTERMEDIATE_ROOT = REPO_ROOT / "LLM_TEST" / "intermediate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build chunk-review groups from chunk_lable data, small-model results, "
            "and full before/after code pairs."
        )
    )
    parser.add_argument(
        "--data_json",
        required=True,
        help="Chunk-level test JSON, e.g. data/chunk_lable/alpaca/chunk_lable_0-512_test.json.",
    )
    parser.add_argument(
        "--results_csv",
        required=True,
        help="Chunk-level model results.csv aligned by Index.",
    )
    parser.add_argument(
        "--pairs_jsonl",
        required=True,
        help="Pairs JSONL containing full before/after method code.",
    )
    parser.add_argument(
        "--prediction_value",
        type=int,
        default=1,
        help="Only keep chunk rows whose small-model prediction equals this value. Defaults to 1.",
    )
    parser.add_argument(
        "--grouping_mode",
        choices=("method", "chunk"),
        default="method",
        help=(
            "How to package review inputs. 'method' groups positive chunks by "
            "(pair_id, side, method_change_id); 'chunk' emits one review item per positive chunk."
        ),
    )
    parser.add_argument(
        "--limit_groups",
        type=int,
        default=None,
        help="Optional limit on how many grouped items to emit.",
    )
    parser.add_argument(
        "--output_root",
        default=str(LLM_TEST_INTERMEDIATE_ROOT),
        help="Root directory for generated review inputs. Defaults to LLM_TEST/intermediate.",
    )
    parser.add_argument(
        "--output_subdir",
        default=None,
        help="Output subdirectory under output_root. Defaults to chunk_lable_chunk_review_<data_json_stem>.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional explicit path for chunk_review_groups.json.",
    )
    parser.add_argument(
        "--positive_chunks_json",
        default=None,
        help="Optional explicit path to save the flat positive chunk rows for inspection.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
        return rows
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def to_int(value: object, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def to_float(value: object, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("pair_id", "")),
        str(row.get("side", "")),
        str(row.get("method_change_id", "")),
    )


def load_results(path: Path) -> dict[int, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = to_int(row.get("Index"), default=None)
        if index is not None:
            mapping[index] = dict(row)
    return mapping


def build_pairs_code_map(path: Path) -> dict[tuple[str, str, str], str]:
    mapping: dict[tuple[str, str, str], str] = {}
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


def derive_output_paths(args: argparse.Namespace, data_json_path: Path) -> tuple[Path, Path, Path, Path]:
    output_root = Path(args.output_root).resolve()
    if args.output_subdir:
        output_subdir = args.output_subdir
    else:
        suffix = "chunk_review_per_chunk" if args.grouping_mode == "chunk" else "chunk_review"
        output_subdir = f"{data_json_path.stem}_{suffix}"
    base_dir = output_root / output_subdir
    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else (base_dir / "chunk_review_groups.json").resolve()
    )
    positive_chunks_json = (
        Path(args.positive_chunks_json).resolve()
        if args.positive_chunks_json
        else (base_dir / "positive_chunks_for_review.json").resolve()
    )
    grouped_results_csv = (base_dir / "grouped_results.csv").resolve()
    summary_json = (base_dir / "summary.json").resolve()
    return output_json, positive_chunks_json, grouped_results_csv, summary_json


def main() -> None:
    args = parse_args()
    data_json_path = Path(args.data_json).resolve()
    results_csv_path = Path(args.results_csv).resolve()
    pairs_jsonl_path = Path(args.pairs_jsonl).resolve()

    data_rows = load_json(data_json_path)
    if not isinstance(data_rows, list):
        raise ValueError("data_json must contain a JSON array.")

    results_by_index = load_results(results_csv_path)
    pairs_code_map = build_pairs_code_map(pairs_jsonl_path)

    positive_chunk_rows: list[dict[str, Any]] = []
    positive_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    missing_results = 0

    for row in data_rows:
        if not isinstance(row, dict):
            continue
        index = to_int(row.get("index"), default=None)
        if index is None:
            continue
        result_row = results_by_index.get(index)
        if result_row is None:
            missing_results += 1
            continue
        prediction = to_int(result_row.get("Prediction"), default=None)
        if prediction != args.prediction_value:
            continue

        merged_row = dict(row)
        merged_row["Prediction"] = prediction
        merged_row["Prob"] = to_float(result_row.get("Prob"), default=None)
        merged_row["Label"] = to_int(result_row.get("Label"), default=None)
        positive_chunk_rows.append(merged_row)
        positive_groups[group_key(row)].append(merged_row)

    group_items: list[dict[str, Any]] = []
    grouped_result_rows: list[dict[str, Any]] = []
    missing_full_code_groups = 0

    if args.grouping_mode == "chunk":
        ordered_rows = sorted(
            positive_chunk_rows,
            key=lambda item: (
                str(item.get("pair_id", "")),
                str(item.get("side", "")),
                str(item.get("method_change_id", "")),
                to_int(item.get("chunk_start_line"), default=10**9),
                to_int(item.get("chunk_id"), default=10**9),
                to_int(item.get("index"), default=10**9),
            ),
        )
        for item_index, row in enumerate(ordered_rows, start=1):
            key = group_key(row)
            full_code = pairs_code_map.get(key)
            if not full_code:
                missing_full_code_groups += 1
                full_code = str(row.get("input", "")).strip()

            review_index = to_int(row.get("index"), default=item_index - 1)
            chunk_item = {
                "chunk_id": str(row.get("chunk_id") or f"chunk-{row.get('index')}"),
                "text": str(row.get("input", "")).strip(),
                "start_line": to_int(row.get("chunk_start_line"), default=None),
                "end_line": to_int(row.get("chunk_end_line"), default=None),
                "prediction": args.prediction_value,
                "vulnerability_probability": to_float(row.get("Prob"), default=None),
                "note": (
                    f"pair_id={row.get('pair_id', '')}; "
                    f"side={row.get('side', '')}; "
                    f"method_change_id={row.get('method_change_id', '')}; "
                    f"sample_index={row.get('index', '')}"
                ),
            }
            group_items.append(
                {
                    "index": review_index,
                    "group_id": f"{row.get('pair_id') or f'group-{item_index}'}::chunk::{row.get('index')}",
                    "sample_id": row.get("pair_id"),
                    "filename": row.get("filename"),
                    "language": row.get("programming_language"),
                    "cwe": row.get("cwe_id", "null"),
                    "ground_truth": to_int(row.get("output", row.get("label")), default=0) or 0,
                    "original_prediction": to_int(row.get("Prediction"), default=0) or 0,
                    "original_probability": to_float(row.get("Prob"), default=None),
                    "code": full_code,
                    "chunks": [chunk_item],
                    "metadata": {
                        "pair_id": row.get("pair_id"),
                        "cve_id": row.get("cve_id"),
                        "cwe_id": row.get("cwe_id"),
                        "commit_hash": row.get("commit_hash"),
                        "file_change_id": row.get("file_change_id"),
                        "change_type": row.get("change_type"),
                        "changed_lines": row.get("changed_lines"),
                        "side": row.get("side"),
                        "method_change_id": row.get("method_change_id"),
                        "method_name": row.get("method_name"),
                        "method_signature": row.get("method_signature"),
                        "chunk_count": 1,
                        "code_source": "pairs_full_code" if pairs_code_map.get(key) else "single_positive_chunk_fallback",
                        "chunk_indexes": [to_int(row.get("index"), default=None)],
                        "index_mapping": "single_chunk_passthrough",
                        "grouping_mode": "chunk",
                    },
                }
            )
            grouped_result_rows.append(
                {
                    "Index": review_index,
                    "Label": to_int(row.get("output", row.get("label")), default=0) or 0,
                    "Prediction": to_int(row.get("Prediction"), default=0) or 0,
                    "Prob": "" if to_float(row.get("Prob"), default=None) is None else to_float(row.get("Prob"), default=None),
                    "CWE": row.get("cwe_id", "null"),
                    "PairId": row.get("pair_id", ""),
                    "Side": row.get("side", ""),
                    "MethodChangeId": row.get("method_change_id", ""),
                    "MethodName": row.get("method_name", ""),
                    "MethodSignature": row.get("method_signature", ""),
                    "ChunkCount": 1,
                    "ChunkIndexes": json.dumps([to_int(row.get("index"), default=None)], ensure_ascii=False),
                }
            )
    else:
        for group_index, key in enumerate(sorted(positive_groups.keys()), start=1):
            rows = sorted(
                positive_groups[key],
                key=lambda item: (
                    to_int(item.get("chunk_start_line"), default=10**9),
                    to_int(item.get("chunk_id"), default=10**9),
                    to_int(item.get("index"), default=10**9),
                ),
            )
            first = rows[0]
            full_code = pairs_code_map.get(key)
            if not full_code:
                missing_full_code_groups += 1
                full_code = "\n".join(str(item.get("input", "")).rstrip("\n") for item in rows if str(item.get("input", "")).strip())

            label_values = [to_int(row.get("output", row.get("label")), default=0) or 0 for row in rows]
            pred_values = [to_int(row.get("Prediction"), default=0) or 0 for row in rows]
            prob_values = [to_float(row.get("Prob"), default=None) for row in rows]
            aggregated_label = max(label_values) if label_values else 0
            aggregated_prediction = max(pred_values) if pred_values else 0
            aggregated_probability = max((value for value in prob_values if value is not None), default=None)

            original_indexes = [to_int(row.get("index"), default=None) for row in rows]
            if len(original_indexes) == 1 and original_indexes[0] is not None:
                review_index = int(original_indexes[0])
                index_mapping = "single_chunk_passthrough"
            else:
                review_index = group_index - 1
                index_mapping = "synthetic_group_index"

            chunk_items: list[dict[str, Any]] = []
            for row in rows:
                chunk_items.append(
                    {
                        "chunk_id": str(row.get("chunk_id") or f"chunk-{row.get('index')}"),
                        "text": str(row.get("input", "")).strip(),
                        "start_line": to_int(row.get("chunk_start_line"), default=None),
                        "end_line": to_int(row.get("chunk_end_line"), default=None),
                        "prediction": args.prediction_value,
                        "vulnerability_probability": to_float(row.get("Prob"), default=None),
                        "note": (
                            f"pair_id={row.get('pair_id', '')}; "
                            f"side={row.get('side', '')}; "
                            f"method_change_id={row.get('method_change_id', '')}; "
                            f"sample_index={row.get('index', '')}"
                        ),
                    }
                )

            group_items.append(
                {
                    "index": review_index,
                    "group_id": str(first.get("pair_id") or f"group-{group_index}"),
                    "sample_id": first.get("pair_id"),
                    "filename": first.get("filename"),
                    "language": first.get("programming_language"),
                    "cwe": first.get("cwe_id", "null"),
                    "ground_truth": aggregated_label,
                    "original_prediction": aggregated_prediction,
                    "original_probability": aggregated_probability,
                    "code": full_code,
                    "chunks": chunk_items,
                    "metadata": {
                        "pair_id": first.get("pair_id"),
                        "cve_id": first.get("cve_id"),
                        "cwe_id": first.get("cwe_id"),
                        "commit_hash": first.get("commit_hash"),
                        "file_change_id": first.get("file_change_id"),
                        "change_type": first.get("change_type"),
                        "changed_lines": first.get("changed_lines"),
                        "side": first.get("side"),
                        "method_change_id": first.get("method_change_id"),
                        "method_name": first.get("method_name"),
                        "method_signature": first.get("method_signature"),
                        "chunk_count": len(chunk_items),
                        "code_source": "pairs_full_code" if pairs_code_map.get(key) else "stitched_positive_chunks",
                        "chunk_indexes": original_indexes,
                        "index_mapping": index_mapping,
                        "grouping_mode": "method",
                    },
                }
            )
            grouped_result_rows.append(
                {
                    "Index": review_index,
                    "Label": aggregated_label,
                    "Prediction": aggregated_prediction,
                    "Prob": "" if aggregated_probability is None else aggregated_probability,
                    "CWE": first.get("cwe_id", "null"),
                    "PairId": first.get("pair_id", ""),
                    "Side": first.get("side", ""),
                    "MethodChangeId": first.get("method_change_id", ""),
                    "MethodName": first.get("method_name", ""),
                    "MethodSignature": first.get("method_signature", ""),
                    "ChunkCount": len(chunk_items),
                    "ChunkIndexes": json.dumps(original_indexes, ensure_ascii=False),
                }
            )

    if args.limit_groups is not None:
        if args.limit_groups <= 0:
            raise ValueError("--limit_groups must be a positive integer.")
        group_items = group_items[: args.limit_groups]
        grouped_result_rows = grouped_result_rows[: args.limit_groups]

    output_json_path, positive_chunks_path, grouped_results_csv_path, summary_json_path = derive_output_paths(args, data_json_path)
    write_json(positive_chunks_path, positive_chunk_rows)
    write_json(
        output_json_path,
        {
            "source_data_json": str(data_json_path),
            "source_results_csv": str(results_csv_path),
            "source_pairs_jsonl": str(pairs_jsonl_path),
            "prediction_value": args.prediction_value,
            "grouping_mode": args.grouping_mode,
            "positive_chunk_count": len(positive_chunk_rows),
            "group_count": len(group_items),
            "missing_results": missing_results,
            "missing_full_code_groups": missing_full_code_groups,
            "grouped_results_csv": str(grouped_results_csv_path),
            "groups": group_items,
        },
    )
    grouped_results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with grouped_results_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
        writer.writeheader()
        writer.writerows(grouped_result_rows)
    write_json(
        summary_json_path,
        {
            "source_data_json": str(data_json_path),
            "source_results_csv": str(results_csv_path),
            "source_pairs_jsonl": str(pairs_jsonl_path),
            "prediction_value": args.prediction_value,
            "grouping_mode": args.grouping_mode,
            "positive_chunk_count": len(positive_chunk_rows),
            "group_count": len(group_items),
            "missing_results": missing_results,
            "missing_full_code_groups": missing_full_code_groups,
            "output_files": {
                "chunk_review_groups_json": str(output_json_path),
                "positive_chunks_json": str(positive_chunks_path),
                "grouped_results_csv": str(grouped_results_csv_path),
            },
        },
    )

    print(f"positive_chunks_json={positive_chunks_path}")
    print(f"output_json={output_json_path}")
    print(f"grouped_results_csv={grouped_results_csv_path}")
    print(f"summary_json={summary_json_path}")
    print(f"positive_chunk_count={len(positive_chunk_rows)}")
    print(f"group_count={len(group_items)}")
    print(f"missing_results={missing_results}")
    print(f"missing_full_code_groups={missing_full_code_groups}")


if __name__ == "__main__":
    main()
