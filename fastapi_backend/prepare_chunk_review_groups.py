from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare batch input for fastapi_backend.chunk_review_cli from existing "
            "frontend code-chunking and code-inference-file outputs."
        )
    )
    parser.add_argument(
        "--chunking_json",
        required=True,
        help="Path to the saved code-chunking result JSON, such as data/user_x/user_x_0-512_test.json.",
    )
    parser.add_argument(
        "--chunking_request_json",
        default=None,
        help="Optional path to chunking_request.json. Defaults to the sibling file next to --chunking_json.",
    )
    parser.add_argument(
        "--inference_json",
        default=None,
        help="Optional path to frontend inference result JSON. It usually already contains only positive chunks.",
    )
    parser.add_argument(
        "--inference_csv",
        default=None,
        help="Optional path to frontend inference result CSV. Use when JSON output is unavailable.",
    )
    parser.add_argument(
        "--prediction_value",
        type=int,
        default=1,
        help="Which prediction value is treated as suspicious and sent to chunk review. Defaults to 1.",
    )
    parser.add_argument(
        "--min_prob",
        type=float,
        default=None,
        help="Optional minimum vulnerability_probability filter.",
    )
    parser.add_argument(
        "--max_groups",
        type=int,
        default=None,
        help="Optional limit on how many grouped review items to emit.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Output JSON path. Defaults to <chunking_json dir>/chunk_review_groups.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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


def build_sample_key(sample_id: object, filename: object) -> tuple[str, str]:
    return str(sample_id or "").strip(), str(filename or "").strip()


def load_full_code_map(chunking_request_payload: Any) -> dict[tuple[str, str], dict[str, Any]]:
    mapping: dict[tuple[str, str], dict[str, Any]] = {}

    if isinstance(chunking_request_payload, dict) and isinstance(chunking_request_payload.get("items"), list):
        items = chunking_request_payload["items"]
    elif isinstance(chunking_request_payload, dict):
        items = [chunking_request_payload]
    else:
        raise ValueError("Unsupported chunking_request.json format.")

    for item_index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip()
        if not code:
            continue
        sample_id = item.get("sample_id")
        filename = item.get("filename")
        sample_key = build_sample_key(sample_id, filename)
        if sample_key == ("", ""):
            sample_key = (f"item-{item_index}", "")
        mapping[sample_key] = {
            "sample_id": sample_id,
            "filename": filename,
            "language": item.get("language"),
            "code": code,
        }
    return mapping


def load_chunking_result_map(chunking_payload: Any) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(chunking_payload, dict) or not isinstance(chunking_payload.get("results"), list):
        raise ValueError("chunking_json must be a saved frontend code-chunking payload with a 'results' list.")

    mapping: dict[tuple[str, str], dict[str, Any]] = {}
    for item_index, result in enumerate(chunking_payload["results"]):
        if not isinstance(result, dict):
            continue
        sample_id = result.get("sample_id")
        filename = result.get("filename")
        sample_key = build_sample_key(sample_id, filename)
        if sample_key == ("", ""):
            sample_key = (f"item-{item_index}", "")
        mapping[sample_key] = result
    return mapping


def load_positive_predictions_from_json(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("inference_json must be a frontend inference result JSON with a 'results' list.")

    rows: list[dict[str, Any]] = []
    for item in payload["results"]:
        if isinstance(item, dict):
            rows.append(dict(item))
    return rows


def load_predictions_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_prediction_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.inference_json:
        return load_positive_predictions_from_json(Path(args.inference_json).resolve())
    if args.inference_csv:
        return load_predictions_from_csv(Path(args.inference_csv).resolve())
    raise ValueError("One of --inference_json or --inference_csv is required.")


def prediction_matches(row: dict[str, Any], prediction_value: int, min_prob: Optional[float]) -> bool:
    prediction = to_int(row.get("prediction", row.get("Prediction")), default=None)
    if prediction is not None and prediction != prediction_value:
        return False

    probability = to_float(row.get("vulnerability_probability", row.get("Prob")), default=None)
    if min_prob is not None and probability is not None and probability < min_prob:
        return False
    if min_prob is not None and probability is None:
        return False
    return True


def derive_output_path(chunking_json: Path, output_json_arg: Optional[str]) -> Path:
    if output_json_arg:
        return Path(output_json_arg).resolve()
    return (chunking_json.parent / "chunk_review_groups.json").resolve()


def main() -> None:
    args = parse_args()
    chunking_json_path = Path(args.chunking_json).resolve()
    if not chunking_json_path.exists():
        raise FileNotFoundError(f"chunking_json not found: {chunking_json_path}")

    chunking_request_path = (
        Path(args.chunking_request_json).resolve()
        if args.chunking_request_json
        else (chunking_json_path.parent / "chunking_request.json").resolve()
    )
    if not chunking_request_path.exists():
        raise FileNotFoundError(
            "chunking_request_json not found. Pass --chunking_request_json explicitly or keep chunking_request.json "
            f"next to {chunking_json_path.name}."
        )

    chunking_payload = load_json(chunking_json_path)
    chunking_request_payload = load_json(chunking_request_path)
    chunking_map = load_chunking_result_map(chunking_payload)
    full_code_map = load_full_code_map(chunking_request_payload)
    prediction_rows = load_prediction_rows(args)

    grouped_chunks: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        if not prediction_matches(row, args.prediction_value, args.min_prob):
            continue
        sample_key = build_sample_key(row.get("sample_id"), row.get("filename"))
        grouped_chunks.setdefault(sample_key, []).append(row)

    groups: list[dict[str, Any]] = []
    unmatched_prediction_rows = 0

    for group_index, sample_key in enumerate(sorted(grouped_chunks.keys()), start=1):
        code_info = full_code_map.get(sample_key)
        chunking_info = chunking_map.get(sample_key)
        if code_info is None or chunking_info is None:
            unmatched_prediction_rows += len(grouped_chunks[sample_key])
            continue

        chunk_lookup: dict[Optional[int], dict[str, Any]] = {}
        for chunk in chunking_info.get("chunks", []):
            if isinstance(chunk, dict):
                chunk_lookup[to_int(chunk.get("index"), default=None)] = chunk

        review_chunks: list[dict[str, Any]] = []
        for row in grouped_chunks[sample_key]:
            chunk_index = to_int(row.get("chunk_index"), default=None)
            chunk_meta = chunk_lookup.get(chunk_index, {})
            chunk_text = str(
                row.get("text")
                or row.get("code")
                or chunk_meta.get("text")
                or ""
            ).strip()
            if not chunk_text:
                unmatched_prediction_rows += 1
                continue

            review_chunks.append(
                {
                    "chunk_id": str(row.get("chunk_id") or f"chunk-{chunk_index if chunk_index is not None else len(review_chunks)}"),
                    "text": chunk_text,
                    "start_line": chunk_meta.get("start_line"),
                    "end_line": chunk_meta.get("end_line"),
                    "prediction": to_int(row.get("prediction", row.get("Prediction")), default=args.prediction_value),
                    "vulnerability_probability": to_float(
                        row.get("vulnerability_probability", row.get("Prob")),
                        default=None,
                    ),
                    "note": (
                        f"sample_id={code_info.get('sample_id') or ''}; "
                        f"filename={code_info.get('filename') or ''}; "
                        f"chunk_index={'' if chunk_index is None else chunk_index}"
                    ),
                }
            )

        if not review_chunks:
            continue

        groups.append(
            {
                "group_id": str(
                    code_info.get("sample_id")
                    or code_info.get("filename")
                    or f"group-{group_index}"
                ),
                "sample_id": code_info.get("sample_id"),
                "filename": code_info.get("filename"),
                "language": code_info.get("language", chunking_info.get("language")),
                "code": code_info["code"],
                "chunks": review_chunks,
            }
        )

    if args.max_groups is not None:
        if args.max_groups <= 0:
            raise ValueError("--max_groups must be a positive integer.")
        groups = groups[: args.max_groups]

    output_path = derive_output_path(chunking_json_path, args.output_json)
    payload = {
        "source_chunking_json": str(chunking_json_path),
        "source_chunking_request_json": str(chunking_request_path),
        "source_inference_json": str(Path(args.inference_json).resolve()) if args.inference_json else None,
        "source_inference_csv": str(Path(args.inference_csv).resolve()) if args.inference_csv else None,
        "prediction_value": args.prediction_value,
        "min_prob": args.min_prob,
        "group_count": len(groups),
        "unmatched_prediction_rows": unmatched_prediction_rows,
        "groups": groups,
    }
    write_json(output_path, payload)

    print(f"output_json={output_path}")
    print(f"group_count={len(groups)}")
    print(f"unmatched_prediction_rows={unmatched_prediction_rows}")


if __name__ == "__main__":
    main()
