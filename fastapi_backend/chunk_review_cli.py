from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from fastapi_backend.frontend_chunk_review import run_frontend_chunk_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run chunk-plus-full-code vulnerability review from the command line. "
            "The input JSON may describe one group or multiple groups."
        )
    )
    parser.add_argument("--input_json", required=True, help="Path to the review input JSON file.")
    parser.add_argument(
        "--output_json",
        default=None,
        help="Path to the aggregated output JSON file. Defaults to LLM_TEST-compatible output_dir/chunk_review_results.json.",
    )
    parser.add_argument("--config", default="LLM_TEST/exp.yaml", help="Path to YAML config.")
    parser.add_argument("--env_file", default="LLM_TEST/.env", help="Path to environment config file.")
    parser.add_argument("--prompt_file", default=None, help="Prompt template file path.")
    parser.add_argument("--model", default=None, help="API model name.")
    parser.add_argument("--api_base", default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api_key", default=None, help="API key. Defaults to env/config.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep_seconds", type=float, default=1.0)
    parser.add_argument(
        "--group_workers",
        type=int,
        default=1,
        help="Number of groups to review concurrently. Defaults to 1.",
    )
    parser.add_argument(
        "--fail_fast_on_error",
        action="store_true",
        help="Exit immediately when one group fails instead of collecting partial results.",
    )
    parser.add_argument(
        "--output_root",
        default="LLM_TEST/output",
        help="Root directory for LLM review outputs. Defaults to LLM_TEST/output.",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help="Output subdirectory under output_root. Defaults to the input JSON parent directory name.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Index",
        "Label",
        "OriginalPrediction",
        "LLMPrediction",
        "LLMModel",
        "ParseStatus",
        "RawResponse",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def derive_output_path(output_dir: Path, output_json_arg: Optional[str]) -> Path:
    if output_json_arg:
        return Path(output_json_arg).resolve()
    return (output_dir / "chunk_review_results.json").resolve()


def normalize_input_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        groups = payload
    elif isinstance(payload, dict) and isinstance(payload.get("groups"), list):
        groups = payload["groups"]
    elif isinstance(payload, dict):
        groups = [payload]
    else:
        raise ValueError("Input JSON must be a group object, a list of groups, or an object with a 'groups' list.")

    if not groups:
        raise ValueError("Input JSON does not contain any review groups.")
    if not all(isinstance(group, dict) for group in groups):
        raise ValueError("Each review group must be a JSON object.")
    return groups


def _load_code(group: dict[str, Any], input_dir: Path) -> str:
    inline_code = group.get("code")
    if isinstance(inline_code, str) and inline_code.strip():
        return inline_code

    code_file = group.get("code_file")
    if isinstance(code_file, str) and code_file.strip():
        code_path = Path(code_file)
        if not code_path.is_absolute():
            code_path = (input_dir / code_path).resolve()
        return code_path.read_text(encoding="utf-8")

    raise ValueError("Each group must provide either a non-empty 'code' or 'code_file'.")


def _load_chunks(group: dict[str, Any], input_dir: Path) -> list[dict[str, Any]]:
    inline_chunks = group.get("chunks")
    if isinstance(inline_chunks, list):
        if not inline_chunks:
            raise ValueError("Each group must contain at least one chunk.")
        if not all(isinstance(chunk, dict) for chunk in inline_chunks):
            raise ValueError("Each chunk must be a JSON object.")
        return inline_chunks

    chunks_file = group.get("chunks_file")
    if isinstance(chunks_file, str) and chunks_file.strip():
        chunks_path = Path(chunks_file)
        if not chunks_path.is_absolute():
            chunks_path = (input_dir / chunks_path).resolve()
        payload = load_json(chunks_path)
        if not isinstance(payload, list) or not payload:
            raise ValueError("'chunks_file' must point to a non-empty JSON array.")
        if not all(isinstance(chunk, dict) for chunk in payload):
            raise ValueError("Each chunk loaded from 'chunks_file' must be a JSON object.")
        return payload

    raise ValueError("Each group must provide either 'chunks' or 'chunks_file'.")


def prepare_group(group: dict[str, Any], group_index: int, input_dir: Path) -> dict[str, Any]:
    prepared = dict(group)
    prepared["group_id"] = str(group.get("group_id") or f"group-{group_index}")
    prepared["index"] = int(group.get("index", group_index - 1))
    prepared["code"] = _load_code(group, input_dir)
    prepared["chunks"] = _load_chunks(group, input_dir)
    return prepared


def review_group(args: argparse.Namespace, group: dict[str, Any]) -> dict[str, Any]:
    result = run_frontend_chunk_review(
        config_path=args.config,
        env_file=args.env_file,
        prompt_file=Path(args.prompt_file).resolve() if args.prompt_file else None,
        code=str(group["code"]),
        chunks=list(group["chunks"]),
        language=group.get("language"),
        model=group.get("model") or args.model,
        api_base=group.get("api_base") or args.api_base,
        api_key=group.get("api_key") or args.api_key,
        temperature=float(group.get("temperature", args.temperature)),
        max_tokens=int(group.get("max_tokens", args.max_tokens)),
        timeout=int(group.get("timeout", args.timeout)),
        retries=int(group.get("retries", args.retries)),
        sleep_seconds=float(group.get("sleep_seconds", args.sleep_seconds)),
    )
    return {
        "index": int(group["index"]),
        "group_id": group["group_id"],
        "language": group.get("language"),
        "chunk_count": len(group["chunks"]),
        "ground_truth": group.get("ground_truth"),
        "original_prediction": group.get("original_prediction"),
        "cwe": group.get("cwe"),
        "result": result,
    }


def build_error_result(group: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "index": int(group["index"]),
        "group_id": group["group_id"],
        "language": group.get("language"),
        "chunk_count": len(group.get("chunks", [])),
        "ground_truth": group.get("ground_truth"),
        "original_prediction": group.get("original_prediction"),
        "cwe": group.get("cwe"),
        "error": f"{type(exc).__name__}: {exc}",
    }


def build_judgment_row(result_item: dict[str, Any], prompt_file: Optional[str]) -> dict[str, Any]:
    review_result = result_item["result"]
    return {
        "dataset_id": "",
        "index": result_item["index"],
        "label": result_item.get("ground_truth"),
        "original_prediction": result_item.get("original_prediction"),
        "llm_prediction": review_result.get("prediction"),
        "model": review_result.get("model", ""),
        "parse_status": review_result.get("parse_status", ""),
        "prompt_file": prompt_file or review_result.get("prompt_file", ""),
        "raw_response": review_result.get("raw_response", ""),
        "group_id": result_item.get("group_id", ""),
        "chunk_count": result_item.get("chunk_count", 0),
        "cwe": result_item.get("cwe", ""),
    }


def build_csv_row(result_item: dict[str, Any]) -> dict[str, Any]:
    review_result = result_item["result"]
    prediction = review_result.get("prediction")
    return {
        "Index": result_item["index"],
        "Label": result_item.get("ground_truth", ""),
        "OriginalPrediction": result_item.get("original_prediction", ""),
        "LLMPrediction": "" if prediction is None else prediction,
        "LLMModel": review_result.get("model", ""),
        "ParseStatus": review_result.get("parse_status", ""),
        "RawResponse": review_result.get("raw_response", ""),
    }


def main() -> None:
    args = parse_args()
    if args.group_workers <= 0:
        raise ValueError("--group_workers must be a positive integer.")

    input_path = Path(args.input_json).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    output_name = args.output_name or input_path.parent.name
    output_dir = Path(args.output_root).resolve() / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = derive_output_path(output_dir, args.output_json)
    raw_payload = load_json(input_path)
    raw_groups = normalize_input_payload(raw_payload)
    prepared_groups = [prepare_group(group, index, input_path.parent) for index, group in enumerate(raw_groups, start=1)]

    results_by_order: list[Optional[dict[str, Any]]] = [None] * len(prepared_groups)

    if args.group_workers == 1:
        for order, group in enumerate(prepared_groups):
            try:
                results_by_order[order] = review_group(args, group)
                print(
                    f"group_id={group['group_id']} status=completed "
                    f"prediction={results_by_order[order]['result']['prediction']} "
                    f"chunks={results_by_order[order]['chunk_count']}"
                )
            except Exception as exc:
                if args.fail_fast_on_error:
                    raise
                results_by_order[order] = build_error_result(group, exc)
                print(f"group_id={group['group_id']} status=failed error={type(exc).__name__}: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=args.group_workers) as executor:
            future_to_order = {
                executor.submit(review_group, args, group): (order, group)
                for order, group in enumerate(prepared_groups)
            }
            for future in as_completed(future_to_order):
                order, group = future_to_order[future]
                try:
                    result = future.result()
                    results_by_order[order] = result
                    print(
                        f"group_id={group['group_id']} status=completed "
                        f"prediction={result['result']['prediction']} "
                        f"chunks={result['chunk_count']}"
                    )
                except Exception as exc:
                    if args.fail_fast_on_error:
                        raise
                    results_by_order[order] = build_error_result(group, exc)
                    print(f"group_id={group['group_id']} status=failed error={type(exc).__name__}: {exc}")

    completed_results = [item for item in results_by_order if item is not None]
    judgment_rows = [build_judgment_row(item, args.prompt_file) for item in completed_results if "result" in item]
    csv_rows = [build_csv_row(item) for item in completed_results if "result" in item]
    payload = {
        "input_json": str(input_path),
        "output_dir": str(output_dir),
        "output_json": str(output_path),
        "output_name": output_name,
        "group_count": len(prepared_groups),
        "completed_groups": sum(1 for item in completed_results if "result" in item),
        "failed_groups": sum(1 for item in completed_results if "error" in item),
        "results": completed_results,
    }
    write_json(output_path, payload)
    write_jsonl(output_dir / "llm_judgments.jsonl", judgment_rows)
    write_csv(output_dir / "llm_predictions.csv", csv_rows)
    write_json(
        output_dir / "llm_summary.json",
        {
            "dataset_id": input_path.parent.name,
            "output_name": output_name,
            "input_json": str(input_path),
            "prompt_file": args.prompt_file,
            "model": "" if not judgment_rows else judgment_rows[0].get("model", ""),
            "total_samples": len(prepared_groups),
            "parsed_predictions": sum(row.get("llm_prediction") in (0, 1) for row in judgment_rows),
            "output_dir": str(output_dir),
            "chunk_review_results_json": str(output_path),
        },
    )
    print(f"output_json={output_path}")
    print(f"output_dir={output_dir}")
    print(f"llm_predictions_csv={output_dir / 'llm_predictions.csv'}")


if __name__ == "__main__":
    main()
