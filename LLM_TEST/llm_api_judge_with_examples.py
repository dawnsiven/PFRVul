import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config_utils import get_section, load_env_file, load_yaml_config, resolve_value
from llm_api_judge import (
    OUTPUT_FORMAT_INSTRUCTIONS,
    build_csv_row,
    build_judgment_row,
    build_output_name,
    call_chat_completion,
    ensure_text,
    extract_text_from_response,
    is_retryable_exception,
    load_json,
    normalize_prompt_template,
    parse_binary_label,
    summarize_error,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call an OpenAI-compatible API and inject a similar historical example into the prompt."
    )
    parser.add_argument("--config", default="LLM_TEST/exp.yaml", help="Path to YAML config.")
    parser.add_argument("--env_file", default="LLM_TEST/.env", help="Path to environment config file.")
    parser.add_argument("--input_json", default=None, help="Path to positive_samples.jsonl.")
    parser.add_argument("--prompt_file", default=None, help="Prompt template with example placeholders.")
    parser.add_argument("--model", default=None, help="API model name.")
    parser.add_argument("--api_base", default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api_key", default=None, help="API key. Defaults to OPENAI_API_KEY.")
    parser.add_argument("--output_root", default=None, help="Root directory for LLM outputs.")
    parser.add_argument("--output_name", default=None, help="Output folder name under output_root.")
    parser.add_argument(
        "--output_by_prompt_version",
        action="store_true",
        help="Use <dataset_id>_<prompt_file_stem> as the output folder name.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N samples.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep_seconds", type=float, default=1.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of samples to process in parallel per batch. Defaults to 4.",
    )
    parser.add_argument(
        "--example_records_json",
        default="LLM_TEST/intermediate/bigvul_cwe20_1_1_train_example_bank.json",
        help="JSON file containing train-split historical cases used as examples.",
    )
    parser.add_argument(
        "--example_similarity_csv",
        default="",
        help="Optional similarity CSV for precomputed nearest neighbors. Leave empty to use on-the-fly retrieval.",
    )
    parser.add_argument(
        "--fail_fast_on_error",
        action="store_true",
        help="Exit immediately when an API or parsing error is encountered.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_code_to_tokens(code: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code.lower())
    return set(tokens)


def jaccard_similarity(code_a: str, code_b: str) -> float:
    tokens_a = normalize_code_to_tokens(code_a)
    tokens_b = normalize_code_to_tokens(code_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union else 0.0


def build_example_map(example_rows: List[dict]) -> Dict[int, dict]:
    return {int(row["Index"]): row for row in example_rows}


def load_example_rows(path: Path) -> List[dict]:
    rows = load_json(path)
    normalized = []
    for row in rows:
        normalized.append(
            {
                "Index": int(row.get("Index", row.get("index", 0))),
                "Label": int(row.get("Label", row.get("ground_truth", 0))),
                "Prediction": int(row.get("Prediction", row.get("original_prediction", 0))),
                "error_type": row.get("error_type", ""),
                "input": row.get("input", ""),
            }
        )
    return normalized


def load_precomputed_neighbors(path: Path) -> Dict[int, List[dict]]:
    if not str(path) or str(path) == "." or not path.exists():
        return {}
    grouped: Dict[int, List[dict]] = {}
    for row in load_csv(path):
        grouped.setdefault(int(row["query_index"]), []).append(row)
    return grouped


def fallback_example_pattern(example_row: dict, similarity: Optional[float]) -> str:
    pattern_parts = [
        f"historical {example_row.get('error_type', 'error')} case",
        f"ground_truth={example_row.get('Label')}",
        f"model_prediction={example_row.get('Prediction')}",
    ]
    if similarity is not None:
        pattern_parts.append(f"lexical_similarity={similarity:.4f}")
    return ", ".join(pattern_parts)


def choose_similar_example(
    sample: dict,
    example_rows: List[dict],
    example_map: Dict[int, dict],
    precomputed_neighbors: Dict[int, List[dict]],
) -> Tuple[dict, str]:
    sample_index = int(sample.get("index", 0))
    sample_code = sample.get("input", "")

    # If the sample already appears in the historical error set, prefer its top precomputed neighbor.
    for neighbor in precomputed_neighbors.get(sample_index, []):
        neighbor_index = int(neighbor["neighbor_index"])
        example_row = example_map.get(neighbor_index)
        if example_row is None:
            continue
        pattern = (
            f"historical {neighbor.get('neighbor_error_type', 'error')} case, "
            f"ground_truth={neighbor.get('neighbor_label', '')}, "
            f"model_prediction={neighbor.get('neighbor_prediction', '')}, "
            f"representation_similarity={float(neighbor.get('similarity', 0.0)):.4f}"
        )
        return example_row, pattern

    best_row = None
    best_similarity = -1.0
    for row in example_rows:
        if int(row["Index"]) == sample_index:
            continue
        similarity = jaccard_similarity(sample_code, row.get("input", ""))
        if similarity > best_similarity:
            best_similarity = similarity
            best_row = row

    if best_row is None:
        raise ValueError("No usable historical example found.")
    return best_row, fallback_example_pattern(best_row, best_similarity)


def build_user_prompt_with_example(prompt_template: str, code: str, example_pattern: str, example_code: str) -> str:
    prompt_template = normalize_prompt_template(prompt_template)
    if "<CODE>" in prompt_template or "<brief pattern>" in prompt_template or "<SIMILAR CODE>" in prompt_template:
        return (
            prompt_template.replace("<CODE>", code)
            .replace("<brief pattern>", example_pattern)
            .replace("<SIMILAR CODE>", example_code)
            + f"\n\n{OUTPUT_FORMAT_INSTRUCTIONS}\n"
        )

    return (
        f"{prompt_template}\n\n"
        "[Target Code]\n"
        f"{code}\n\n"
        "A similar historical case (previously misclassified by a model) is provided:\n"
        f"- Pattern: {example_pattern}\n"
        "- Snippet:\n"
        f"{example_code}\n"
        f"\n{OUTPUT_FORMAT_INSTRUCTIONS}\n"
    )


def process_sample_with_example(
    sample: dict,
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt_template: str,
    prompt_file: Path,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    sleep_seconds: float,
    example_rows: List[dict],
    example_map: Dict[int, dict],
    precomputed_neighbors: Dict[int, List[dict]],
) -> Tuple[dict, dict]:
    example_row, example_pattern = choose_similar_example(
        sample=sample,
        example_rows=example_rows,
        example_map=example_map,
        precomputed_neighbors=precomputed_neighbors,
    )
    user_prompt = build_user_prompt_with_example(
        prompt_template=prompt_template,
        code=sample.get("input", ""),
        example_pattern=example_pattern,
        example_code=example_row.get("input", ""),
    )

    response_text = ""
    parse_status = "unparsed"
    llm_prediction: Optional[int] = None

    for attempt in range(1, retries + 1):
        try:
            payload = call_chat_completion(
                api_base=api_base,
                api_key=api_key,
                model=model,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            response_text = extract_text_from_response(payload)
            llm_prediction, parse_source = parse_binary_label(response_text)
            parse_status = parse_source or "unparsed"
            if llm_prediction is None:
                raise ValueError("Could not parse vulnerable=yes/no from model response.")
            break
        except Exception as exc:  # noqa: BLE001
            last_error = summarize_error(exc, response_text)
            should_retry = is_retryable_exception(exc)
            if attempt == retries or not should_retry:
                response_text = response_text or last_error or ""
                parse_status = f"error:{type(exc).__name__}"
                llm_prediction = None
            else:
                import time

                time.sleep(sleep_seconds)

    judgment = build_judgment_row(
        sample=sample,
        model=model,
        prompt_file=prompt_file,
        llm_prediction=llm_prediction,
        parse_status=parse_status,
        response_text=response_text,
    )
    judgment["example_index"] = example_row["Index"]
    judgment["example_pattern"] = example_pattern
    judgment["example_code"] = example_row.get("input", "")

    csv_row = build_csv_row(sample, model, llm_prediction, parse_status, response_text)
    csv_row["ExampleIndex"] = example_row["Index"]
    csv_row["ExamplePattern"] = example_pattern
    return judgment, csv_row


def write_csv_with_examples(path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "Index",
        "Label",
        "OriginalPrediction",
        "LLMPrediction",
        "LLMModel",
        "ParseStatus",
        "RawResponse",
        "ExampleIndex",
        "ExamplePattern",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    config = load_yaml_config(args.config)
    common_cfg = get_section(config, "common")
    llm_cfg = get_section(config, "llm")

    input_json_value = resolve_value(args.input_json, llm_cfg, "input_json")
    prompt_file_value = resolve_value(args.prompt_file, llm_cfg, "prompt_file")
    model = resolve_value(args.model, llm_cfg, "model")
    api_base = resolve_value(args.api_base, llm_cfg, "api_base")
    api_key_env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = resolve_value(args.api_key, llm_cfg, "api_key")
    if not api_key:
        import os

        api_key = os.environ.get(api_key_env)
    output_root = resolve_value(args.output_root, common_cfg, "output_root", "LLM_TEST/output")
    temperature = float(resolve_value(args.temperature, llm_cfg, "temperature", 0.0))
    max_tokens = int(resolve_value(args.max_tokens, llm_cfg, "max_tokens", 256))
    timeout = int(resolve_value(args.timeout, llm_cfg, "timeout", 120))
    retries = int(resolve_value(args.retries, llm_cfg, "retries", 3))
    sleep_seconds = float(resolve_value(args.sleep_seconds, llm_cfg, "sleep_seconds", 1.0))
    workers = int(resolve_value(args.workers, llm_cfg, "workers", 4))

    if workers <= 0:
        raise ValueError("--workers must be a positive integer.")

    if not api_key:
        raise ValueError(
            f"API key is required. Please set '{api_key_env}' in {Path(args.env_file).resolve()} "
            "or pass --api_key."
        )

    input_json = Path(input_json_value).resolve()
    prompt_file = Path(prompt_file_value).resolve()
    example_records_json = Path(args.example_records_json).resolve()
    example_similarity_csv = Path(args.example_similarity_csv).resolve() if args.example_similarity_csv else Path()

    positive_samples = load_json(input_json)
    prompt_template = ensure_text(prompt_file)
    example_rows = load_example_rows(example_records_json)
    example_map = build_example_map(example_rows)
    precomputed_neighbors = load_precomputed_neighbors(example_similarity_csv)

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer.")
        positive_samples = positive_samples[: args.limit]

    dataset_id = input_json.parent.name
    output_name = build_output_name(
        dataset_id=dataset_id,
        prompt_file=prompt_file,
        output_name_arg=args.output_name,
        output_by_prompt_version=args.output_by_prompt_version,
    )
    output_dir = Path(output_root).resolve() / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    judgment_rows: List[dict] = []
    csv_rows: List[dict] = []

    for batch_start in range(0, len(positive_samples), workers):
        batch = positive_samples[batch_start : batch_start + workers]
        ordered_results: List[Optional[Tuple[dict, dict]]] = [None] * len(batch)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_pos = {
                executor.submit(
                    process_sample_with_example,
                    sample,
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    prompt_template=prompt_template,
                    prompt_file=prompt_file,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    retries=retries,
                    sleep_seconds=sleep_seconds,
                    example_rows=example_rows,
                    example_map=example_map,
                    precomputed_neighbors=precomputed_neighbors,
                ): pos
                for pos, sample in enumerate(batch)
            }

            for future in as_completed(future_to_pos):
                pos = future_to_pos[future]
                ordered_results[pos] = future.result()

        for sample, result in zip(batch, ordered_results):
            if result is None:
                raise RuntimeError(f"Missing result for sample index={sample['index']}")
            judgment, csv_row = result
            judgment["dataset_id"] = dataset_id
            judgment_rows.append(judgment)
            csv_rows.append(csv_row)

            print(
                f"index={sample['index']} label={sample['ground_truth']} "
                f"original={sample['original_prediction']} llm={judgment['llm_prediction']} "
                f"example={judgment['example_index']} status={judgment['parse_status']}"
            )
            if judgment["parse_status"].startswith("error:"):
                print(f"error_detail={judgment['raw_response']}")
            if args.fail_fast_on_error and judgment["parse_status"].startswith("error:"):
                raise RuntimeError(
                    f"Fail-fast triggered at index={sample['index']} with status={judgment['parse_status']}: "
                    f"{judgment['raw_response']}"
                )

        write_jsonl(output_dir / "llm_judgments.jsonl", judgment_rows)
        write_csv_with_examples(output_dir / "llm_predictions.csv", csv_rows)

    write_jsonl(output_dir / "llm_judgments.jsonl", judgment_rows)
    write_csv_with_examples(output_dir / "llm_predictions.csv", csv_rows)
    write_json(
        output_dir / "llm_summary.json",
        {
            "dataset_id": dataset_id,
            "input_json": str(input_json),
            "prompt_file": str(prompt_file),
            "model": model,
            "api_base": api_base,
            "example_records_json": str(example_records_json),
            "example_similarity_csv": str(example_similarity_csv) if str(example_similarity_csv) else "",
            "workers": workers,
            "processed_samples": len(judgment_rows),
            "parsed_predictions": sum(1 for row in judgment_rows if row["llm_prediction"] is not None),
            "error_predictions": sum(1 for row in judgment_rows if row["parse_status"].startswith("error:")),
        },
    )


if __name__ == "__main__":
    main()
