import argparse
import csv
import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config_utils import get_section, load_env_file, load_yaml_config, resolve_value


OUTPUT_FORMAT_INSTRUCTIONS = """Output requirements:
- Return exactly one JSON object.
- Use this schema:
{
  "vulnerable": "yes/no",
  "cwe": "...",
  "source": "...",
  "sink": "...",
  "missing_check": "...",
  "reason": "..."
}
- Do not wrap the JSON in Markdown fences.
- Do not add any text before or after the JSON."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call an OpenAI-compatible API to re-check extracted positive samples."
    )
    parser.add_argument("--config", default="LLM_TEST/exp.yaml", help="Path to YAML config.")
    parser.add_argument("--env_file", default="LLM_TEST/.env", help="Path to environment config file.")
    parser.add_argument(
        "--input_json",
        default=None,
        help="Path to LLM_TEST/intermediate/<dataset_id>/positive_samples.jsonl",
    )
    parser.add_argument(
        "--prompt_file",
        default=None,
        help="Prompt template file. The code snippet will be appended automatically.",
    )
    parser.add_argument("--model", default=None, help="API model name.")
    parser.add_argument(
        "--api_base",
        default=None,
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Root directory for LLM outputs.",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help="Output folder name under output_root. Defaults to dataset_id for backward compatibility.",
    )
    parser.add_argument(
        "--output_by_prompt_version",
        action="store_true",
        help="Use <dataset_id>_<prompt_file_stem> as the output folder name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples from the input JSON.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum output tokens. When omitted, do not send max_tokens to the API.",
    )
    parser.add_argument(
        "--response_format",
        choices=("text", "json_object", "json_schema"),
        default="text",
        help="Response format passed to OpenAI-compatible APIs.",
    )
    parser.add_argument(
        "--json_schema_file",
        default=None,
        help="Path to a JSON file used when --response_format json_schema.",
    )
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep_seconds", type=float, default=1.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent API requests. Defaults to 1 (serial).",
    )
    parser.add_argument(
        "--fail_fast_on_error",
        action="store_true",
        help="Exit immediately when an API or parsing error is encountered.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Only process samples whose sample['index'] is greater than or equal to this value.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing outputs in the target directory and skip only successfully completed sample indexes.",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


def ensure_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read().strip()


def load_json_object(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload).__name__}.")
    return payload


def normalize_prompt_template(prompt_template: str) -> str:
    text = prompt_template.strip()
    format_markers = [
        r"\nOutput in JSON format:\s*\{.*?\}\s*$",
        r"\nReturn only 0 or 1\.?\s*$",
        r"\nOutput requirements:\s*\{.*?\}\s*$",
    ]
    for pattern in format_markers:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def build_user_prompt(prompt_template: str, code: str) -> str:
    prompt_template = normalize_prompt_template(prompt_template)
    return (
        f"{prompt_template}\n\n"
        "Code:\n"
        "```c\n"
        f"{code}\n"
        "```\n"
        f"\n{OUTPUT_FORMAT_INSTRUCTIONS}\n"
    )


def extract_text_segments(content: object) -> List[str]:
    if isinstance(content, str):
        return [content.strip()] if content.strip() else []
    if not isinstance(content, list):
        return []

    text_parts: List[str] = []
    for item in content:
        if isinstance(item, dict):
            item_type = str(item.get("type", "")).strip().lower()
            if item_type in {"reasoning", "thinking", "reasoning_content"}:
                continue
            if isinstance(item.get("text"), str) and item["text"].strip():
                text_parts.append(item["text"].strip())
                continue
            if item_type in {"text", "output_text"} and isinstance(item.get("content"), str):
                if item["content"].strip():
                    text_parts.append(item["content"].strip())
        elif isinstance(item, str) and item.strip():
            text_parts.append(item.strip())
    return text_parts


def extract_text_from_response(payload: dict) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""

    first = choices[0]
    message = first.get("message", {})
    if isinstance(message, dict):
        text_parts = extract_text_segments(message.get("content"))
        if text_parts:
            return "\n".join(text_parts)

    text = first.get("text")
    return str(text).strip() if text is not None else ""


def extract_json_candidates(text: str) -> List[str]:
    candidates: List[str] = []

    fenced_matches = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced_matches)

    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text[start:])
            candidates.append(text[start : start + end])
        except json.JSONDecodeError:
            continue
    return candidates


def try_parse_json_block(text: str) -> Optional[dict]:
    candidates = extract_json_candidates(text)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def select_final_response_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""

    candidates = extract_json_candidates(stripped)
    for candidate in reversed(candidates):
        try:
            json.loads(candidate)
            return candidate.strip()
        except json.JSONDecodeError:
            continue

    non_empty_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if non_empty_lines:
        last_line = non_empty_lines[-1]
        if re.fullmatch(r"(?i)yes|true|1|no|false|0", last_line):
            return last_line

    return stripped


def parse_binary_label(text: str) -> Tuple[Optional[int], Optional[str]]:
    parsed = try_parse_json_block(text)
    if isinstance(parsed, dict):
        vulnerable = str(parsed.get("vulnerable", "")).strip().lower()
        if vulnerable in {"yes", "true", "1"}:
            return 1, "json.vulnerable"
        if vulnerable in {"no", "false", "0"}:
            return 0, "json.vulnerable"

    lowered = text.lower()
    if re.search(r'"vulnerable"\s*:\s*"?(yes|true|1)"?', lowered):
        return 1, "regex.vulnerable"
    if re.search(r'"vulnerable"\s*:\s*"?(no|false|0)"?', lowered):
        return 0, "regex.vulnerable"

    stripped = text.strip()
    if re.fullmatch(r"(?i)yes|true|1", stripped):
        return 1, "exact.yes"
    if re.fullmatch(r"(?i)no|false|0", stripped):
        return 0, "exact.no"

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if non_empty_lines:
        last_line = non_empty_lines[-1]
        if re.fullmatch(r"(?i)yes|true|1", last_line):
            return 1, "last_line.yes"
        if re.fullmatch(r"(?i)no|false|0", last_line):
            return 0, "last_line.no"
    return None, None


def is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, http.client.RemoteDisconnected):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, ValueError):
        message = str(exc)
        if re.match(r"^HTTP 429\b", message):
            return True
        if re.match(r"^HTTP 4\d\d\b", message):
            return False
    return True


def call_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: Optional[int],
    response_format: Optional[dict],
    timeout: int,
) -> dict:
    url = api_base.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "stream": False,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful vulnerability detection assistant. Return valid JSON whenever possible.",
            },
            {"role": "user", "content": user_prompt},
        ],
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if response_format is not None:
        body["response_format"] = response_format

    request = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace").strip()
        detail = f"HTTP {exc.code}"
        if error_body:
            detail = f"{detail}: {error_body}"
        raise ValueError(detail) from exc


def summarize_error(exc: Exception, response_text: str) -> str:
    detail = str(exc).strip()
    if response_text.strip():
        compact_response = re.sub(r"\s+", " ", response_text).strip()
        if len(compact_response) > 200:
            compact_response = compact_response[:200] + "..."
        if detail:
            return f"{detail} | response={compact_response}"
        return f"response={compact_response}"
    return detail or type(exc).__name__


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[dict]) -> None:
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
        for row in rows:
            writer.writerow(row)


def to_int(value: object, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def load_existing_csv_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def index_existing_csv_rows(rows: List[dict]) -> Dict[int, dict]:
    indexed: Dict[int, dict] = {}
    for row in rows:
        sample_index = to_int(row.get("Index"))
        if sample_index is None:
            continue
        indexed[sample_index] = row
    return indexed


def index_existing_judgments(rows: List[dict]) -> Dict[int, dict]:
    indexed: Dict[int, dict] = {}
    for row in rows:
        sample_index = to_int(row.get("index"))
        if sample_index is None:
            continue
        indexed[sample_index] = row
    return indexed


def is_successful_judgment_row(row: Optional[dict]) -> bool:
    if not row:
        return False
    parse_status = str(row.get("parse_status", "")).strip()
    llm_prediction = to_int(row.get("llm_prediction"))
    return not parse_status.startswith("error:") and llm_prediction in (0, 1)


def is_successful_csv_row(row: Optional[dict]) -> bool:
    if not row:
        return False
    parse_status = str(row.get("ParseStatus", "")).strip()
    llm_prediction = to_int(row.get("LLMPrediction"))
    return not parse_status.startswith("error:") and llm_prediction in (0, 1)


def build_output_name(
    dataset_id: str,
    prompt_file: Path,
    output_name_arg: Optional[str],
    output_by_prompt_version: bool,
) -> str:
    if output_name_arg:
        return output_name_arg
    if output_by_prompt_version:
        return f"{dataset_id}_{prompt_file.stem}"
    return dataset_id


def build_judgment_row(
    sample: dict,
    model: str,
    prompt_file: Path,
    llm_prediction: Optional[int],
    parse_status: str,
    response_text: str,
) -> dict:
    return {
        "dataset_id": "",
        "index": sample["index"],
        "label": sample["ground_truth"],
        "original_prediction": sample["original_prediction"],
        "llm_prediction": llm_prediction,
        "model": model,
        "parse_status": parse_status,
        "prompt_file": str(prompt_file),
        "raw_response": response_text,
    }


def build_csv_row(sample: dict, model: str, llm_prediction: Optional[int], parse_status: str, response_text: str) -> dict:
    return {
        "Index": sample["index"],
        "Label": sample["ground_truth"],
        "OriginalPrediction": sample["original_prediction"],
        "LLMPrediction": "" if llm_prediction is None else llm_prediction,
        "LLMModel": model,
        "ParseStatus": parse_status,
        "RawResponse": response_text,
    }


def process_sample(
    sample: dict,
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt_template: str,
    prompt_file: Path,
    temperature: float,
    max_tokens: Optional[int],
    response_format: Optional[dict],
    timeout: int,
    retries: int,
    sleep_seconds: float,
) -> Tuple[dict, dict]:
    user_prompt = build_user_prompt(prompt_template, sample.get("input", ""))

    response_text = ""
    raw_response_text = ""
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
                response_format=response_format,
                timeout=timeout,
            )
            raw_response_text = extract_text_from_response(payload)
            response_text = select_final_response_text(raw_response_text)
            llm_prediction, parse_source = parse_binary_label(response_text)
            parse_status = parse_source or "unparsed"
            if llm_prediction is None:
                raise ValueError("Could not parse vulnerable=yes/no from model response.")
            break
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            TimeoutError,
            ValueError,
        ) as exc:
            last_error = summarize_error(exc, raw_response_text or response_text)
            should_retry = is_retryable_exception(exc)
            if attempt == retries or not should_retry:
                response_text = response_text or raw_response_text or last_error or ""
                parse_status = f"error:{type(exc).__name__}"
                llm_prediction = None
            else:
                time.sleep(sleep_seconds)

    judgment = build_judgment_row(
        sample=sample,
        model=model,
        prompt_file=prompt_file,
        llm_prediction=llm_prediction,
        parse_status=parse_status,
        response_text=response_text,
    )
    csv_row = build_csv_row(sample, model, llm_prediction, parse_status, response_text)
    return judgment, csv_row


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    config = load_yaml_config(args.config)
    common_cfg = get_section(config, "common")
    llm_cfg = get_section(config, "llm")

    input_json_value = resolve_value(args.input_json, llm_cfg, "input_json")
    prompt_file_value = resolve_value(args.prompt_file, llm_cfg, "prompt_file")
    model = resolve_value(args.model, llm_cfg, "model")
    api_base = resolve_value(
        args.api_base,
        llm_cfg,
        "api_base",
        os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com/v1"),
    )
    api_key_env = llm_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = resolve_value(args.api_key, llm_cfg, "api_key", os.environ.get(api_key_env))
    output_root = resolve_value(args.output_root, common_cfg, "output_root", "LLM_TEST/output")
    temperature = float(resolve_value(args.temperature, llm_cfg, "temperature", 0.0))
    max_tokens_value = resolve_value(args.max_tokens, llm_cfg, "max_tokens")
    max_tokens = int(max_tokens_value) if max_tokens_value not in (None, "") else None
    response_format_type = str(resolve_value(args.response_format, llm_cfg, "response_format", "text"))
    timeout = int(resolve_value(args.timeout, llm_cfg, "timeout", 120))
    retries = int(resolve_value(args.retries, llm_cfg, "retries", 3))
    sleep_seconds = float(resolve_value(args.sleep_seconds, llm_cfg, "sleep_seconds", 1.0))
    response_format: Optional[dict] = None

    if response_format_type == "json_object":
        response_format = {"type": "json_object"}
    elif response_format_type == "json_schema":
        json_schema_file_value = resolve_value(args.json_schema_file, llm_cfg, "json_schema_file")
        if not json_schema_file_value:
            raise ValueError("--json_schema_file is required when --response_format json_schema.")
        json_schema_file = Path(json_schema_file_value).resolve()
        if not json_schema_file.exists():
            raise FileNotFoundError(f"JSON schema file not found: {json_schema_file}")
        response_format = {
            "type": "json_schema",
            "json_schema": load_json_object(json_schema_file),
        }
    elif response_format_type != "text":
        raise ValueError(f"Unsupported response_format: {response_format_type}")

    if not api_key:
        raise ValueError(
            f"API key is required. Please set '{api_key_env}' in {Path(args.env_file).resolve()} "
            "or pass --api_key."
        )

    input_json = Path(input_json_value).resolve()
    prompt_file = Path(prompt_file_value).resolve()
    if not input_json.exists():
        raise FileNotFoundError(
            "Input JSON not found: "
            f"{input_json}. "
            "Run LLM_TEST/extract_positive_samples.py first, or fix INPUT_JSON / --input_json."
        )
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    positive_samples = load_json(input_json)
    prompt_template = ensure_text(prompt_file)
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

    if args.workers <= 0:
        raise ValueError("--workers must be a positive integer.")

    sample_order_by_index = {
        int(sample["index"]): order for order, sample in enumerate(positive_samples) if sample.get("index") is not None
    }
    judgments_by_order: List[Optional[dict]] = [None] * len(positive_samples)
    csv_by_order: List[Optional[dict]] = [None] * len(positive_samples)
    judgments_path = output_dir / "llm_judgments.jsonl"
    csv_path = output_dir / "llm_predictions.csv"

    if args.resume:
        existing_judgments = load_json(judgments_path) if judgments_path.exists() else []
        existing_csv_rows = load_existing_csv_rows(csv_path)
        judgments_by_index = index_existing_judgments(existing_judgments)
        csv_by_index = index_existing_csv_rows(existing_csv_rows)

        for sample_index, order in sample_order_by_index.items():
            judgment = judgments_by_index.get(sample_index)
            csv_row = csv_by_index.get(sample_index)
            if judgment is not None:
                judgment.setdefault("dataset_id", dataset_id)
                judgments_by_order[order] = judgment
            if csv_row is not None:
                csv_by_order[order] = csv_row

    pending_orders: List[int] = []
    for order, sample in enumerate(positive_samples):
        sample_index = int(sample["index"])
        if args.start_index is not None and sample_index < args.start_index:
            continue
        if (
            args.resume
            and is_successful_judgment_row(judgments_by_order[order])
            and is_successful_csv_row(csv_by_order[order])
        ):
            continue
        pending_orders.append(order)

    print(f"pending_samples={len(pending_orders)}")

    if args.workers == 1:
        for order in pending_orders:
            sample = positive_samples[order]
            judgment, csv_row = process_sample(
                sample,
                api_base=api_base,
                api_key=api_key,
                model=model,
                prompt_template=prompt_template,
                prompt_file=prompt_file,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout=timeout,
                retries=retries,
                sleep_seconds=sleep_seconds,
            )
            judgment["dataset_id"] = dataset_id
            judgments_by_order[order] = judgment
            csv_by_order[order] = csv_row

            print(
                f"index={sample['index']} label={sample['ground_truth']} "
                f"original={sample['original_prediction']} llm={judgment['llm_prediction']} "
                f"status={judgment['parse_status']}"
            )
            if judgment["parse_status"].startswith("error:"):
                print(f"error_detail={judgment['raw_response']}")
            if args.fail_fast_on_error and judgment["parse_status"].startswith("error:"):
                raise RuntimeError(
                    f"Fail-fast triggered at index={sample['index']} with status={judgment['parse_status']}: "
                    f"{judgment['raw_response']}"
                )

            write_jsonl(judgments_path, [row for row in judgments_by_order if row is not None])
            write_csv(csv_path, [row for row in csv_by_order if row is not None])
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_order = {
                executor.submit(
                    process_sample,
                    positive_samples[order],
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    prompt_template=prompt_template,
                    prompt_file=prompt_file,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    timeout=timeout,
                    retries=retries,
                    sleep_seconds=sleep_seconds,
                ): (order, positive_samples[order])
                for order in pending_orders
            }

            for future in as_completed(future_to_order):
                order, sample = future_to_order[future]
                judgment, csv_row = future.result()
                judgment["dataset_id"] = dataset_id
                judgments_by_order[order] = judgment
                csv_by_order[order] = csv_row

                print(
                    f"index={sample['index']} label={sample['ground_truth']} "
                    f"original={sample['original_prediction']} llm={judgment['llm_prediction']} "
                    f"status={judgment['parse_status']}"
                )
                if judgment["parse_status"].startswith("error:"):
                    print(f"error_detail={judgment['raw_response']}")
                if args.fail_fast_on_error and judgment["parse_status"].startswith("error:"):
                    raise RuntimeError(
                        f"Fail-fast triggered at index={sample['index']} with status={judgment['parse_status']}: "
                        f"{judgment['raw_response']}"
                    )

                completed_judgments = [row for row in judgments_by_order if row is not None]
                completed_csv_rows = [row for row in csv_by_order if row is not None]
                write_jsonl(judgments_path, completed_judgments)
                write_csv(csv_path, completed_csv_rows)

    judgment_rows = [row for row in judgments_by_order if row is not None]
    csv_rows = [row for row in csv_by_order if row is not None]

    write_json(
        output_dir / "llm_summary.json",
        {
            "dataset_id": dataset_id,
            "output_name": output_name,
            "input_json": str(input_json),
            "prompt_file": str(prompt_file),
            "model": model,
            "response_format": response_format_type,
            "total_samples": len(positive_samples),
            "parsed_predictions": sum(row["llm_prediction"] in (0, 1) for row in judgment_rows),
            "output_dir": str(output_dir),
        },
    )

    print(f"dataset_id={dataset_id}")
    print(f"output_name={output_name}")
    print(f"total_samples={len(positive_samples)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
