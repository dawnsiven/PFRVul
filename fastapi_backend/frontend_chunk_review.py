from __future__ import annotations

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


OUTPUT_FORMAT_INSTRUCTIONS = """Output requirements:
- Return exactly one JSON object.
- Use this schema:
{
  "vulnerable": "yes/no",
  "cwe": "...",
  "reason": "..."
}
- If a field is unknown, use an empty string.
- Do not wrap the JSON in Markdown fences.
- Do not add any text before or after the JSON."""

ENV_VAR_PATTERN = re.compile(r"\$(?:\{([^}]+)\}|([A-Za-z_][A-Za-z0-9_]*))")


def load_env_file(env_path: Optional[str]) -> None:
    if not env_path:
        return

    path = Path(env_path).resolve()
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def _expand_env_in_string_loose(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1) or match.group(2) or ""
        return os.environ.get(env_name, "")

    return ENV_VAR_PATTERN.sub(replace, value)


def expand_env_vars_loose(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: expand_env_vars_loose(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars_loose(item) for item in value]
    if isinstance(value, str):
        return _expand_env_in_string_loose(value) if "$" in value else value
    return value


def load_yaml_config_loose(config_path: Optional[str]) -> dict[str, Any]:
    if not config_path:
        return {}

    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read the review config.") from exc

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Top-level YAML content must be a mapping: {path}")
    return expand_env_vars_loose(payload)


def get_section(config: dict[str, Any], section: str) -> dict[str, Any]:
    value = config.get(section, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section}' must be a mapping.")
    return value


def resolve_value(cli_value: Any, section: dict[str, Any], key: str, default: Any = None) -> Any:
    if cli_value not in (None, ""):
        return cli_value
    if key in section and section[key] not in (None, ""):
        return section[key]
    return default


def ensure_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def normalize_prompt_template(prompt_template: str) -> str:
    return prompt_template.strip()


def _normalize_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_id = str(chunk.get("chunk_id") or f"chunk-{index}")
        normalized.append(
            {
                "chunk_id": chunk_id,
                "text": str(chunk.get("text") or "").strip(),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "prediction": chunk.get("prediction"),
                "vulnerability_probability": chunk.get("vulnerability_probability"),
                "note": chunk.get("note"),
            }
        )
    return normalized


def _format_chunk(chunk: dict[str, Any], language: Optional[str]) -> str:
    metadata: list[str] = [f"chunk_id: {chunk['chunk_id']}"]

    if chunk.get("start_line") is not None or chunk.get("end_line") is not None:
        start_line = chunk.get("start_line", "")
        end_line = chunk.get("end_line", "")
        metadata.append(f"lines: {start_line}-{end_line}")
    if chunk.get("note"):
        metadata.append(f"note: {chunk['note']}")

    fence = (language or "").strip()
    metadata_block = "\n".join(f"- {item}" for item in metadata)
    return (
        f"{metadata_block}\n"
        "chunk_code:\n"
        f"```{fence}\n"
        f"{chunk['text']}\n"
        "```"
    )


def build_user_prompt(
    prompt_template: str,
    code: str,
    chunk: dict[str, Any],
    language: Optional[str],
) -> str:
    prompt_template = normalize_prompt_template(prompt_template)
    fence = (language or "").strip()
    return (
        f"{prompt_template}\n\n"
        "The following code chunk was flagged as vulnerable by the small model. "
        "Please judge whether this chunk is truly vulnerable under the full original code context.\n\n"
        "Suspicious chunk:\n"
        f"{_format_chunk(chunk, language)}\n\n"
        "Original full code:\n"
        f"```{fence}\n"
        f"{code}\n"
        "```\n\n"
        f"{OUTPUT_FORMAT_INSTRUCTIONS}\n"
    )


def extract_text_from_response(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""

    first = choices[0]
    message = first.get("message", {})
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str) and item["text"].strip():
                        text_parts.append(item["text"].strip())
                    elif item.get("type") == "text" and isinstance(item.get("content"), str):
                        if item["content"].strip():
                            text_parts.append(item["content"].strip())
                elif isinstance(item, str) and item.strip():
                    text_parts.append(item.strip())
            if text_parts:
                return "\n".join(text_parts)

    text = first.get("text")
    return str(text).strip() if text is not None else ""


def try_parse_json_block(text: str) -> Optional[dict[str, Any]]:
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []

    braces = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if braces:
        candidates.append(braces.group(1))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_binary_label(text: str) -> tuple[Optional[int], Optional[str], Optional[dict[str, Any]]]:
    parsed = try_parse_json_block(text)
    if isinstance(parsed, dict):
        vulnerable = str(parsed.get("vulnerable", "")).strip().lower()
        if vulnerable in {"yes", "true", "1"}:
            return 1, "json.vulnerable", parsed
        if vulnerable in {"no", "false", "0"}:
            return 0, "json.vulnerable", parsed

    lowered = text.lower()
    if re.search(r'"vulnerable"\s*:\s*"?(yes|true|1)"?', lowered):
        return 1, "regex.vulnerable", parsed
    if re.search(r'"vulnerable"\s*:\s*"?(no|false|0)"?', lowered):
        return 0, "regex.vulnerable", parsed
    return None, None, parsed


def is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, http.client.RemoteDisconnected):
        return True
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, ValueError) and re.match(r"^HTTP 4\d\d\b", str(exc)):
        return False
    return True


def call_chat_completion(
    api_base: str,
    api_key: str,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    url = api_base.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful vulnerability detection assistant. Return valid JSON whenever possible.",
            },
            {"role": "user", "content": user_prompt},
        ],
    }

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


def build_chunk_verdict(
    *,
    chunk: dict[str, Any],
    prediction: Optional[int],
    parse_status: str,
    parsed_response: Optional[dict[str, Any]],
    raw_response: str,
) -> dict[str, Any]:
    vulnerable_value = None
    cwe = ""
    reason = ""
    if isinstance(parsed_response, dict):
        raw_vulnerable = str(parsed_response.get("vulnerable", "")).strip().lower()
        if raw_vulnerable in {"yes", "no"}:
            vulnerable_value = raw_vulnerable
        cwe = str(parsed_response.get("cwe", "") or "")
        reason = str(parsed_response.get("reason", "") or "")

    return {
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "prediction": prediction,
        "vulnerable": vulnerable_value,
        "cwe": cwe,
        "reason": reason,
        "parse_status": parse_status,
        "start_line": chunk.get("start_line"),
        "end_line": chunk.get("end_line"),
        "raw_response": raw_response,
    }


def review_single_chunk(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt_template: str,
    code: str,
    chunk: dict[str, Any],
    language: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    sleep_seconds: float,
) -> tuple[Optional[int], str, Optional[dict[str, Any]], str]:
    user_prompt = build_user_prompt(prompt_template, code, chunk, language)
    response_text = ""
    parse_status = "unparsed"
    prediction: Optional[int] = None
    parsed_response: Optional[dict[str, Any]] = None

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
            prediction, parse_source, parsed_response = parse_binary_label(response_text)
            parse_status = parse_source or "unparsed"
            if prediction is None:
                raise ValueError("Could not parse vulnerable=yes/no from model response.")
            return prediction, parse_status, parsed_response, response_text
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.RemoteDisconnected,
            TimeoutError,
            ValueError,
        ) as exc:
            last_error = summarize_error(exc, response_text)
            if attempt == retries or not is_retryable_exception(exc):
                response_text = response_text or last_error or ""
                parse_status = f"error:{type(exc).__name__}"
                return None, parse_status, parsed_response, response_text
            time.sleep(sleep_seconds)

    return prediction, parse_status, parsed_response, response_text


def run_frontend_chunk_review(
    *,
    config_path: str,
    env_file: str,
    prompt_file: Optional[Path],
    code: str,
    chunks: list[dict[str, Any]],
    language: Optional[str],
    model: Optional[str],
    api_base: Optional[str],
    api_key: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    load_env_file(env_file)
    config = load_yaml_config_loose(config_path)
    llm_cfg = get_section(config, "llm")

    prompt_file_value = resolve_value(str(prompt_file) if prompt_file is not None else None, llm_cfg, "prompt_file")
    model_value = resolve_value(model, llm_cfg, "model")
    api_base_value = resolve_value(
        api_base,
        llm_cfg,
        "api_base",
        os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com/v1"),
    )
    api_key_env = str(llm_cfg.get("api_key_env", "OPENAI_API_KEY") or "OPENAI_API_KEY")
    api_key_value = resolve_value(api_key, llm_cfg, "api_key", os.environ.get(api_key_env))

    if not prompt_file_value:
        raise ValueError("Prompt file is required. Provide prompt_file or configure llm.prompt_file.")
    if not model_value:
        raise ValueError("Model is required. Provide model or configure llm.model.")
    if not api_key_value:
        raise ValueError(
            f"API key is required. Set '{api_key_env}' in the environment, in {Path(env_file).resolve()}, or pass api_key."
        )

    resolved_prompt_file = Path(prompt_file_value).resolve()
    if not resolved_prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {resolved_prompt_file}")

    normalized_chunks = _normalize_chunks(chunks)
    prompt_template = ensure_text(resolved_prompt_file)
    chunk_verdicts: list[dict[str, Any]] = []
    aggregated_raw_responses: list[str] = []
    prediction: Optional[int] = 0
    parse_status = "ok"
    parsed_response: Optional[dict[str, Any]] = None

    for chunk in normalized_chunks:
        chunk_prediction, chunk_parse_status, chunk_parsed_response, chunk_raw_response = review_single_chunk(
            api_base=str(api_base_value),
            api_key=str(api_key_value),
            model=str(model_value),
            prompt_template=prompt_template,
            code=code,
            chunk=chunk,
            language=language,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            sleep_seconds=sleep_seconds,
        )
        chunk_verdicts.append(
            build_chunk_verdict(
                chunk=chunk,
                prediction=chunk_prediction,
                parse_status=chunk_parse_status,
                parsed_response=chunk_parsed_response,
                raw_response=chunk_raw_response,
            )
        )
        aggregated_raw_responses.append(f"[{chunk.get('chunk_id')}] {chunk_raw_response}")

        if chunk_parse_status.startswith("error:") and parse_status == "ok":
            parse_status = chunk_parse_status
        if chunk_prediction is None:
            prediction = None
        elif prediction is not None and chunk_prediction == 1:
            prediction = 1

    if len(chunk_verdicts) == 1:
        parsed_response = {
            "vulnerable": chunk_verdicts[0].get("vulnerable"),
            "cwe": chunk_verdicts[0].get("cwe"),
            "reason": chunk_verdicts[0].get("reason"),
        }
        if parse_status == "ok":
            parse_status = str(chunk_verdicts[0].get("parse_status") or "ok")
    else:
        if parse_status == "ok":
            statuses = [str(item.get("parse_status") or "ok") for item in chunk_verdicts]
            parse_status = "ok" if all(status == "json.vulnerable" for status in statuses) else "partial"

    return {
        "model": str(model_value),
        "prompt_file": str(resolved_prompt_file),
        "prediction": prediction,
        "is_vulnerable": None if prediction is None else bool(prediction == 1),
        "parse_status": parse_status,
        "parsed_response": parsed_response,
        "chunk_verdicts": chunk_verdicts,
        "raw_response": "\n\n".join(aggregated_raw_responses),
        "chunk_count": len(normalized_chunks),
    }
