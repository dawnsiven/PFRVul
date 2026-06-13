from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from chunk_splite.build_cvefixes_astchunk import build_chunks, normalize_language, token_count

from fastapi_backend.job_runner import REPO_ROOT


DEFAULT_TOKENIZER_MODEL = REPO_ROOT / "model" / "Llama-3.2-1B"
DATA_ROOT = REPO_ROOT / "data"


def normalize_frontend_code(code: str) -> str:
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")

    # Some frontend callers submit already-escaped code blobs such as "\\n" and "\\t"
    # instead of real control characters. In that case chunking sees a single long line.
    if "\\n" not in normalized:
        return normalized

    if "\n" in normalized:
        return normalized

    try:
        unescaped = bytes(normalized, "utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return normalized

    if "\n" not in unescaped:
        return normalized

    return unescaped.replace("\r\n", "\n").replace("\r", "\n")


def _resolve_tokenizer_model_path(path_value: Optional[str]) -> Optional[str]:
    if path_value is None:
        default_path = DEFAULT_TOKENIZER_MODEL
        return str(default_path) if default_path.exists() else None

    cleaned = path_value.strip()
    if not cleaned:
        return None

    candidate = Path(cleaned)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return str(candidate.resolve())


def run_code_chunking(
    *,
    code: str,
    language: Optional[str] = None,
    max_chars: int = 1800,
    max_tokens: int = 512,
    fallback_lines: int = 80,
    tokenizer_model_path: Optional[str] = None,
) -> dict[str, object]:
    code = normalize_frontend_code(code)
    resolved_tokenizer_model_path = _resolve_tokenizer_model_path(tokenizer_model_path)
    normalized_language = normalize_language(language)
    chunks, chunk_source = build_chunks(
        code=code,
        language=language,
        max_chars=max_chars,
        max_tokens=max_tokens,
        fallback_lines=fallback_lines,
        tokenizer_model_path=resolved_tokenizer_model_path,
    )

    chunk_payloads: list[dict[str, object]] = []
    for index, chunk in enumerate(chunks):
        chunk_payloads.append(
            {
                "index": index,
                "text": chunk.text,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "token_count": token_count(chunk.text, resolved_tokenizer_model_path),
            }
        )

    return {
        "language": language,
        "normalized_language": normalized_language,
        "chunk_source": chunk_source,
        "max_chars": max_chars,
        "max_tokens": max_tokens,
        "fallback_lines": fallback_lines,
        "tokenizer_model_path": resolved_tokenizer_model_path,
        "chunk_count": len(chunk_payloads),
        "chunks": chunk_payloads,
    }


def build_user_chunking_dir(folder_name: str) -> Path:
    cleaned = folder_name.strip()
    if not cleaned:
        raise ValueError("folder_name cannot be empty.")

    candidate = Path(cleaned)
    if candidate.is_absolute() or candidate.name != cleaned or cleaned in {".", ".."}:
        raise ValueError("folder_name must be a simple directory name.")

    target = (DATA_ROOT / f"user_{cleaned}").resolve()
    try:
        target.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Invalid folder_name.") from exc

    target.mkdir(parents=True, exist_ok=True)
    return target


def save_chunking_payload(
    *,
    folder_name: str,
    max_tokens: int,
    payload: dict[str, object],
    request_payload: Optional[dict[str, object]] = None,
) -> Path:
    target_dir = build_user_chunking_dir(folder_name)
    persisted_payload = dict(payload)
    persisted_payload["storage_dir"] = str(target_dir)
    dataset_stem = f"user_{folder_name.strip()}_0-{max_tokens}_test"

    with (target_dir / f"{dataset_stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(persisted_payload, handle, ensure_ascii=False, indent=2)

    with (target_dir / "chunking_examples.json").open("w", encoding="utf-8") as handle:
        json.dump(persisted_payload.get("examples", []), handle, ensure_ascii=False, indent=2)

    if request_payload is not None:
        with (target_dir / "chunking_request.json").open("w", encoding="utf-8") as handle:
            json.dump(request_payload, handle, ensure_ascii=False, indent=2)

    return target_dir
