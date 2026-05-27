from __future__ import annotations

import mimetypes
import os
import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from fastapi_backend.chunk_review_schemas import (
    FrontendChunkReviewRequest,
    FrontendChunkReviewResponse,
)
from fastapi_backend.code_chunking import run_code_chunking, save_chunking_payload
from fastapi_backend.code_inference import run_file_inference, run_single_inference
from fastapi_backend.frontend_chunk_review import run_frontend_chunk_review
from fastapi_backend.job_runner import JobManager, REPO_ROOT
from fastapi_backend.schemas import (
    AblationJobRequest,
    ClassicalJobRequest,
    FrontendCodeChunkExample,
    FrontendCodeChunkFlat,
    FrontendCodeChunkItem,
    FrontendCodeChunkRequest,
    FrontendCodeChunkResponse,
    FrontendCodeChunkResult,
    DataUploadResponse,
    FrontendCodeInferenceRequest,
    FrontendCodeInferenceResponse,
    FrontendCodeInferenceFileRequest,
    FrontendCodeInferenceFileResponse,
    FrontendDataIngestResponse,
    FrontendFindCodeRequest,
    FrontendFindCodeResponse,
    FileOption,
    FrontendInputJsonOptionsResponse,
    ImbalanceClassicalJobRequest,
    ImbalanceLLMJobRequest,
    JobLogResponse,
    JobResponse,
    LLMTestEnsembleJobRequest,
    LLMTestExtractJobRequest,
    LLMTestFullMetricsJobRequest,
    LLMTestFullPrepareJobRequest,
    LLMTestJudgeJobRequest,
    LLMTestMetricsJobRequest,
    LLMTestOptionsResponse,
    LLMTestReviewJobRequest,
    LLMJobRequest,
    OutputEntry,
    OutputListResponse,
    OutputTextResponse,
    PromptCreateRequest,
    PromptFileResponse,
    ToGraphJobRequest,
)

app = FastAPI(
    title="LLM4CVD FastAPI Backend",
    version="0.1.0",
    description="HTTP API for launching the existing training and inference scripts in this repository.",
)

origins = [item.strip() for item in os.getenv("ALLOWED_ORIGINS", "*").split(",") if item.strip()]
allow_credentials = origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

job_manager = JobManager()
DATA_ROOT = REPO_ROOT / "data"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
LLM_TEST_ROOT = REPO_ROOT / "LLM_TEST"
LLM_TEST_PROMPT_ROOT = LLM_TEST_ROOT / "Prompt"
LLM_TEST_INTERMEDIATE_ROOT = LLM_TEST_ROOT / "intermediate"
LLM_TEST_OUTPUT_ROOT = LLM_TEST_ROOT / "output"
LLM_TEST_ALLOWED_ROOTS = {
    "prompt": LLM_TEST_PROMPT_ROOT,
    "intermediate": LLM_TEST_INTERMEDIATE_ROOT,
    "output": LLM_TEST_OUTPUT_ROOT,
}

GRAPH_MODELS = {"Devign", "ReGVD", "GraphCodeBERT", "CodeBERT", "UniXcoder"}
LLM_MODELS = {"llama2", "llama3", "llama3.1", "codellama","llama3.2"}


def _job_response(record) -> JobResponse:
    return JobResponse(**record.as_dict())


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{label} not found: {path}")


def _tail_text(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(content[-lines:])


def _resolve_outputs_path(relative_path: str = "") -> Path:
    normalized_relative = relative_path.strip().lstrip("/")
    if normalized_relative == "outputs":
        normalized_relative = ""
    elif normalized_relative.startswith("outputs/"):
        normalized_relative = normalized_relative[len("outputs/") :]

    candidate = (OUTPUTS_ROOT / normalized_relative).resolve()
    try:
        candidate.relative_to(OUTPUTS_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid outputs path.") from exc
    return candidate


def _resolve_data_path(relative_path: str = "") -> Path:
    normalized_relative = relative_path.strip().lstrip("/")
    if normalized_relative == "data":
        normalized_relative = ""
    elif normalized_relative.startswith("data/"):
        normalized_relative = normalized_relative[len("data/") :]

    candidate = (DATA_ROOT / normalized_relative).resolve()
    try:
        candidate.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid data path.") from exc
    return candidate


def _resolve_user_scoped_path(base_root: Path, relative_path: str, label: str) -> Path:
    normalized_relative = relative_path.strip().lstrip("/")
    if not normalized_relative:
        return base_root

    if str(base_root.name) == "data" and normalized_relative.startswith("data/"):
        normalized_relative = normalized_relative[len("data/") :]
    elif str(base_root.name) == "outputs" and normalized_relative.startswith("outputs/"):
        normalized_relative = normalized_relative[len("outputs/") :]

    candidate = (base_root / normalized_relative).resolve()
    try:
        candidate.relative_to(base_root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid {label} path.") from exc

    relative_parts = candidate.relative_to(base_root.resolve()).parts
    if relative_parts and not relative_parts[0].startswith("user_"):
        raise HTTPException(status_code=400, detail=f"{label} path must stay inside a user_* folder.")
    return candidate


def _resolve_repo_path(path_value: str, label: str, *, allow_empty: bool = False) -> Optional[Path]:
    cleaned = path_value.strip()
    if not cleaned:
        if allow_empty:
            return None
        raise HTTPException(status_code=400, detail=f"{label} cannot be empty.")

    candidate = Path(cleaned)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()

    try:
        candidate.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{label} must stay inside the repository.") from exc
    return candidate


def _strip_subsampled_test_suffix(file_stem: str) -> str:
    if not file_stem.endswith("_test"):
        return file_stem
    prefix = file_stem[: -len("_test")]
    parts = prefix.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:-1]) + "_test"
    return file_stem


def _resolve_full_prepare_data_json(path_value: str) -> Path:
    candidate = _resolve_repo_path(path_value, "data_json")
    if candidate.exists():
        parts = candidate.relative_to(REPO_ROOT).parts
        if len(parts) >= 4 and parts[0] == "data" and parts[1].endswith("_subsampled") and parts[-1].endswith("_test.json"):
            base_dataset = parts[1][: -len("_subsampled")]
            normalized_name = _strip_subsampled_test_suffix(candidate.stem) + candidate.suffix
            fallback = (REPO_ROOT / "data" / base_dataset / "alpaca" / normalized_name).resolve()
            if fallback.exists():
                return fallback
        return candidate

    relative_parts = candidate.relative_to(REPO_ROOT).parts
    if len(relative_parts) >= 3 and relative_parts[0] == "data":
        dataset_dir = relative_parts[1]
        filename = relative_parts[-1]
        fallback_candidates: list[Path] = []

        fallback_candidates.append((REPO_ROOT / "data" / dataset_dir / "alpaca" / filename).resolve())

        if dataset_dir.endswith("_subsampled") and filename.endswith("_test.json"):
            base_dataset = dataset_dir[: -len("_subsampled")]
            normalized_name = _strip_subsampled_test_suffix(Path(filename).stem) + Path(filename).suffix
            fallback_candidates.append((REPO_ROOT / "data" / base_dataset / "alpaca" / normalized_name).resolve())

        for fallback in fallback_candidates:
            try:
                fallback.relative_to(REPO_ROOT.resolve())
            except ValueError:
                continue
            if fallback.exists():
                return fallback

    raise HTTPException(status_code=404, detail=f"data_json not found: {_to_relative_repo_path(candidate)}")


def _resolve_checkpoint_dir(path_value: str) -> Path:
    cleaned = path_value.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="checkpoint_dir cannot be empty.")

    if Path(cleaned).is_absolute():
        checkpoint_dir = _resolve_repo_path(cleaned, "checkpoint_dir")
    else:
        # Frontend checkpoint options come from /api/outputs, so relative paths
        # should resolve under outputs/ whether they include the prefix or not.
        checkpoint_dir = _resolve_outputs_path(cleaned)

    # Accept callers that accidentally pass the checkpoint file or its parent.
    if checkpoint_dir.name == "model.bin":
        checkpoint_dir = checkpoint_dir.parent.parent
    elif checkpoint_dir.name == "checkpoint-best-f1":
        checkpoint_dir = checkpoint_dir.parent

    candidate_dirs: list[Path] = []

    def _add_candidate(path: Path) -> None:
        if path not in candidate_dirs:
            candidate_dirs.append(path)

    _add_candidate(checkpoint_dir)

    # Some callers append "_imbalance" to the dataset directory name.
    if checkpoint_dir.name.endswith("_imbalance"):
        _add_candidate(checkpoint_dir.with_name(checkpoint_dir.name[: -len("_imbalance")]))

    parent_name = checkpoint_dir.parent.name
    dataset_name = checkpoint_dir.name
    if parent_name and checkpoint_dir.parent.parent == OUTPUTS_ROOT:
        # Real imbalance checkpoints live under outputs/<MODEL>_imbalance/<DATASET>_<POS_RATIO>.
        imbalance_root = checkpoint_dir.parent.with_name(f"{parent_name}_imbalance")
        if dataset_name.endswith("_imbalance"):
            base_dataset_name = dataset_name[: -len("_imbalance")]
            _add_candidate(imbalance_root / dataset_name)
            _add_candidate(imbalance_root / base_dataset_name)
            _add_candidate(imbalance_root / f"{base_dataset_name}_1")
        else:
            _add_candidate(imbalance_root / dataset_name)
            _add_candidate(imbalance_root / f"{dataset_name}_1")

    for candidate_dir in candidate_dirs:
        checkpoint_file = candidate_dir / "checkpoint-best-f1" / "model.bin"
        if checkpoint_file.exists():
            return candidate_dir

    searched_paths = ", ".join(str(path / "checkpoint-best-f1" / "model.bin") for path in candidate_dirs)
    raise HTTPException(status_code=404, detail=f"checkpoint file not found: {searched_paths}")


def _resolve_llm_test_path(root_key: str, relative_path: str = "") -> Path:
    base_root = LLM_TEST_ALLOWED_ROOTS.get(root_key)
    if base_root is None:
        raise HTTPException(status_code=400, detail="Unsupported LLM_TEST root.")

    candidate = (base_root / relative_path).resolve()
    try:
        candidate.relative_to(base_root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid LLM_TEST path.") from exc
    return candidate


def _read_text_file(path: Path, max_chars: int) -> str:
    content = path.read_text(encoding="utf-8", errors="ignore")
    if len(content) > max_chars:
        return content[:max_chars]
    return content


def _to_relative_repo_path(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _normalize_prompt_name(prompt_name: str) -> str:
    cleaned = prompt_name.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="prompt_name cannot be empty.")

    candidate = Path(cleaned)
    if candidate.is_absolute() or candidate.name != cleaned or cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail="prompt_name must be a simple file name.")

    if candidate.suffix != ".txt":
        cleaned = f"{cleaned}.txt"
    return cleaned


def _normalize_upload_filename(filename: str) -> str:
    cleaned = Path(filename.strip()).name
    if not cleaned or cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid file name.")
    if Path(cleaned).name != cleaned:
        raise HTTPException(status_code=400, detail="Uploaded file name must not contain path separators.")
    return cleaned


def _normalize_dataset_name(dataset_name: str) -> str:
    cleaned = dataset_name.strip()
    if not cleaned:
        raise HTTPException(status_code=400, detail="dataset_name cannot be empty.")
    candidate = Path(cleaned)
    if candidate.is_absolute() or candidate.name != cleaned or cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail="dataset_name must be a simple directory name.")
    return cleaned


def _parse_bucket_boundaries(bucket_boundaries: str) -> list[int]:
    values: list[int] = []
    for item in bucket_boundaries.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid bucket boundary: {stripped}") from exc
        if parsed <= 0:
            raise HTTPException(status_code=400, detail="Bucket boundaries must be positive integers.")
        values.append(parsed)
    if not values:
        raise HTTPException(status_code=400, detail="At least one bucket boundary is required.")
    return sorted(set(values))


def _build_bucket_specs(boundaries: list[int]) -> list[tuple[str, int, Optional[int]]]:
    bucket_specs: list[tuple[str, int, Optional[int]]] = []
    lower = 0
    for upper in boundaries:
        bucket_specs.append((f"{lower}-{upper}", lower, upper))
        lower = upper
    bucket_specs.append((f"{lower}-*", lower, None))
    return bucket_specs


def _find_bucket(length: int, bucket_specs: list[tuple[str, int, Optional[int]]]) -> str:
    for bucket_name, lower, upper in bucket_specs:
        if upper is None and length >= lower:
            return bucket_name
        if upper is not None and lower <= length < upper:
            return bucket_name
    raise HTTPException(status_code=400, detail=f"Unable to bucket sample with token length {length}.")


def _load_uploaded_json_array(path: Path) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Uploaded file is not valid JSON: {path.name}") from exc
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="Uploaded JSON must be a top-level array.")
    if any(not isinstance(item, dict) for item in payload):
        raise HTTPException(status_code=400, detail="Uploaded JSON array must contain only objects.")
    return payload


def _detect_frontend_data_format(rows: list[dict]) -> str:
    if not rows:
        raise HTTPException(status_code=400, detail="Uploaded JSON array is empty.")
    sample = rows[0]
    if "code" in sample and "label" in sample:
        return "raw"
    if "input" in sample and "output" in sample:
        return "alpaca"
    raise HTTPException(
        status_code=400,
        detail="Unsupported JSON schema. Expected raw {code,label,index} or alpaca {instruction,input,output,index}.",
    )


def _to_alpaca_rows(rows: list[dict], input_format: str, instruction: str) -> list[dict]:
    alpaca_rows: list[dict] = []
    for position, row in enumerate(rows):
        if input_format == "raw":
            if "code" not in row or "label" not in row:
                raise HTTPException(status_code=400, detail=f"Raw row {position} is missing code or label.")
            alpaca_rows.append(
                {
                    "instruction": instruction,
                    "input": row.get("code", ""),
                    "output": str(row.get("label", "")),
                    "index": row.get("index", position),
                }
            )
        else:
            if "input" not in row or "output" not in row:
                raise HTTPException(status_code=400, detail=f"Alpaca row {position} is missing input or output.")
            normalized = dict(row)
            normalized.setdefault("instruction", instruction)
            normalized.setdefault("index", position)
            normalized["output"] = str(normalized.get("output", ""))
            alpaca_rows.append(normalized)
    return alpaca_rows


def _create_prompt_file(prompt_name: str, prompt_content: str, *, overwrite: bool) -> Path:
    content = prompt_content.strip()
    if not content:
        raise HTTPException(status_code=400, detail="prompt_content cannot be empty.")

    LLM_TEST_PROMPT_ROOT.mkdir(parents=True, exist_ok=True)
    normalized_name = _normalize_prompt_name(prompt_name)
    target = (LLM_TEST_PROMPT_ROOT / normalized_name).resolve()
    try:
        target.relative_to(LLM_TEST_PROMPT_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid prompt target path.") from exc

    if target.exists() and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Prompt already exists: {_to_relative_repo_path(target)}. Set overwrite=true to replace it.",
        )

    target.write_text(prompt_content, encoding="utf-8")
    return target


def _resolve_or_create_prompt_file(
    *,
    prompt_file: Optional[str],
    prompt_name: Optional[str],
    prompt_content: Optional[str],
    prompt_overwrite: bool,
) -> Path:
    has_prompt_file = bool(prompt_file and prompt_file.strip())
    has_prompt_name = bool(prompt_name and prompt_name.strip())
    has_prompt_content = bool(prompt_content and prompt_content.strip())

    if has_prompt_file and (has_prompt_name or has_prompt_content):
        raise HTTPException(
            status_code=400,
            detail="Use either prompt_file or prompt_name + prompt_content, not both.",
        )

    if has_prompt_file:
        return _resolve_repo_path(prompt_file, "prompt_file")

    if has_prompt_name or has_prompt_content:
        if not has_prompt_name or not has_prompt_content:
            raise HTTPException(
                status_code=400,
                detail="prompt_name and prompt_content must be provided together.",
            )
        return _create_prompt_file(prompt_name or "", prompt_content or "", overwrite=prompt_overwrite)

    raise HTTPException(
        status_code=400,
        detail="Either prompt_file or prompt_name + prompt_content is required.",
    )


def _build_file_option(path: Path) -> FileOption:
    relative_path = _to_relative_repo_path(path)
    return FileOption(label=path.parent.name + "/" + path.name, path=relative_path)


def _scan_glob_options(base_root: Path, pattern: str) -> list[FileOption]:
    if not base_root.exists():
        return []
    return [_build_file_option(path) for path in sorted(base_root.glob(pattern))]


def _scan_dir_options(base_root: Path, child_name: str) -> list[FileOption]:
    if not base_root.exists():
        return []

    options: list[FileOption] = []
    for path in sorted(base_root.glob(f"**/{child_name}")):
        if path.parent.is_dir():
            options.append(
                FileOption(
                    label=path.parent.name,
                    path=_to_relative_repo_path(path.parent),
                )
            )
    return options


def _merge_file_options(*option_groups: list[FileOption]) -> list[FileOption]:
    merged: list[FileOption] = []
    seen_paths: set[str] = set()
    for group in option_groups:
        for item in group:
            if item.path in seen_paths:
                continue
                seen_paths.add(item.path)
                merged.append(item)
    return merged


def _scan_frontend_input_json_options() -> list[FileOption]:
    if not DATA_ROOT.exists():
        return []

    options: list[FileOption] = []
    for user_dir in sorted(DATA_ROOT.iterdir()):
        if not user_dir.is_dir() or not user_dir.name.startswith("user_"):
            continue
        for path in sorted(user_dir.glob("*.json")):
            options.append(
                FileOption(
                    label=path.name,
                    path=_to_relative_repo_path(path),
                )
            )
    return options


def _list_user_root_directories(base_root: Path) -> OutputListResponse:
    entries: list[OutputEntry] = []
    if base_root.exists():
        for item in sorted(base_root.iterdir(), key=lambda path: path.name.lower()):
            if not item.is_dir() or not item.name.startswith("user_"):
                continue
            entries.append(
                OutputEntry(
                    name=item.name,
                    relative_path=item.name,
                    entry_type="directory",
                    size=None,
                )
            )
    return OutputListResponse(base_dir=str(base_root), relative_path="", entries=entries)


def _find_code_from_chunking_request(
    *,
    relative_path: str,
    sample_id: Optional[str],
    filename: Optional[str],
) -> FrontendFindCodeResponse:
    target = _resolve_user_scoped_path(DATA_ROOT, relative_path, "data")
    _assert_exists(target, "data file")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Target path is not a file.")
    if target.name != "chunking_request.json":
        raise HTTPException(status_code=400, detail="relative_path must point to chunking_request.json.")

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="chunking_request.json is not valid JSON.") from exc

    items = payload.get("items")
    if not isinstance(items, list):
        raise HTTPException(status_code=400, detail="chunking_request.json does not contain a valid items array.")

    normalized_sample_id = (sample_id or "").strip()
    normalized_filename = (filename or "").strip()
    if not normalized_sample_id and not normalized_filename:
        raise HTTPException(status_code=400, detail="Provide sample_id or filename.")

    match: Optional[dict] = None
    if normalized_sample_id:
        for item in items:
            if isinstance(item, dict) and str(item.get("sample_id", "")).strip() == normalized_sample_id:
                match = item
                break
    elif normalized_filename:
        for item in items:
            if isinstance(item, dict) and str(item.get("filename", "")).strip() == normalized_filename:
                match = item
                break

    if match is None:
        query_label = f"sample_id={normalized_sample_id}" if normalized_sample_id else f"filename={normalized_filename}"
        raise HTTPException(status_code=404, detail=f"No matching item found for {query_label}.")

    code = str(match.get("code") or "").strip()
    if not code:
        raise HTTPException(status_code=404, detail="Matched item does not contain code.")

    return FrontendFindCodeResponse(
        sample_id=None if match.get("sample_id") is None else str(match.get("sample_id")),
        filename=None if match.get("filename") is None else str(match.get("filename")),
        language=None if match.get("language") is None else str(match.get("language")),
        code=code,
    )


def _list_directory(root_path: Path, target: Path) -> OutputListResponse:
    entries = []
    for item in sorted(target.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())):
        resolved = item.resolve()
        entries.append(
            OutputEntry(
                name=item.name,
                relative_path=str(resolved.relative_to(root_path.resolve())),
                entry_type="directory" if item.is_dir() else "file",
                size=None if item.is_dir() else item.stat().st_size,
            )
        )

    normalized_relative = "" if target == root_path else str(target.relative_to(root_path.resolve()))
    return OutputListResponse(
        base_dir=str(root_path),
        relative_path=normalized_relative,
        entries=entries,
    )


def _scan_data_directories() -> dict[str, list[str]]:
    data_root = REPO_ROOT / "data"
    regular = []
    imbalance = []
    if data_root.exists():
        for child in sorted(item.name for item in data_root.iterdir() if item.is_dir()):
            if child.endswith("_subsampled"):
                imbalance.append(child)
            else:
                regular.append(child)
    return {"regular": regular, "imbalance": imbalance}


def _collect_suffixes(directory: Path, prefix: str, suffix: str) -> list[str]:
    if not directory.exists():
        return []
    values = set()
    for file_path in directory.glob(f"{prefix}*{suffix}"):
        stem = file_path.name
        if not stem.startswith(prefix) or not stem.endswith(suffix):
            continue
        values.add(stem[len(prefix) : -len(suffix)])
    return sorted(values)


def _dataset_prefix_from_directory(directory: Path) -> str:
    dataset_name = directory.parent.name
    if dataset_name.endswith("_subsampled"):
        dataset_name = dataset_name[: -len("_subsampled")]
    return f"{dataset_name}_"


def _collect_imbalance_parts(directory: Path) -> tuple[list[str], list[str]]:
    lengths = set()
    ratios = set()
    if not directory.exists():
        return [], []

    prefix = _dataset_prefix_from_directory(directory)
    suffix = "_train.json"
    for item in _collect_suffixes(directory, prefix, suffix):
        parts = item.split("_")
        if len(parts) >= 2 and "-" in parts[0]:
            lengths.add(parts[0])
            ratios.add("_".join(parts[1:]))
        elif item:
            ratios.add(item)
    return sorted(lengths), sorted(ratios)


def _stringify_params(payload: dict[str, object]) -> dict[str, str]:
    return {key: str(value) for key, value in payload.items() if value is not None}


def _launch_command_job(
    *,
    job_type: str,
    script_name: str,
    command: list[str],
    output_dir: Path | None,
    log_file: Path | None,
    result_csv: Path | None,
    params: dict[str, object],
):
    record = job_manager.start_job(
        job_type=job_type,
        script_name=script_name,
        command=command,
        params=_stringify_params(params),
        output_dir=output_dir,
        log_file=log_file,
        result_csv=result_csv,
        working_dir=REPO_ROOT,
    )
    return _job_response(record)


def _launch_job(
    *,
    job_type: str,
    script_name: str,
    args: Iterable[object],
    output_dir: Path | None,
    log_file: Path | None,
    result_csv: Path | None,
    params: dict[str, object],
):
    script_path = REPO_ROOT / "scripts" / script_name
    _assert_exists(script_path, "script")
    command = ["bash", str(script_path), *[str(arg) for arg in args]]
    record = job_manager.start_job(
        job_type=job_type,
        script_name=script_name,
        command=command,
        params=_stringify_params(params),
        output_dir=output_dir,
        log_file=log_file,
        result_csv=result_csv,
        working_dir=REPO_ROOT,
    )
    return _job_response(record)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/data/upload", response_model=DataUploadResponse)
def upload_data_file(
    file: UploadFile = File(...),
    relative_dir: str = Form(default=""),
    overwrite: bool = Form(default=False),
) -> DataUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a file name.")

    filename = _normalize_upload_filename(file.filename)
    target_dir = _resolve_data_path(relative_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = (target_dir / filename).resolve()
    try:
        target_path.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid upload target path.") from exc

    if target_path.exists() and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Data file already exists: {_to_relative_repo_path(target_path)}. Set overwrite=true to replace it.",
        )

    try:
        with target_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
    finally:
        file.file.close()

    return DataUploadResponse(
        path=_to_relative_repo_path(target_path),
        relative_path=str(target_path.relative_to(DATA_ROOT.resolve())),
        filename=target_path.name,
        size=target_path.stat().st_size,
    )


@app.post("/api/frontend/data/ingest", response_model=FrontendDataIngestResponse)
def frontend_data_ingest(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    split_name: str = Form(default="test"),
    overwrite: bool = Form(default=False),
    bucket_boundaries: str = Form(default="512,1024"),
    tokenizer_path: str = Form(default=""),
    instruction: str = Form(default="Detect whether the following code contains vulnerabilities."),
) -> FrontendDataIngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must include a file name.")

    normalized_dataset_name = _normalize_dataset_name(dataset_name)
    normalized_split_name = split_name.strip()
    if normalized_split_name not in {"train", "validate", "test"}:
        raise HTTPException(status_code=400, detail="split_name must be one of train, validate, or test.")

    boundaries = _parse_bucket_boundaries(bucket_boundaries)
    bucket_specs = _build_bucket_specs(boundaries)

    dataset_dir = _resolve_data_path(normalized_dataset_name)
    uploads_dir = dataset_dir / "uploads"
    alpaca_dir = dataset_dir / "alpaca"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    alpaca_dir.mkdir(parents=True, exist_ok=True)

    filename = _normalize_upload_filename(file.filename)
    uploaded_path = (uploads_dir / filename).resolve()
    try:
        uploaded_path.relative_to(DATA_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid upload target path.") from exc

    if uploaded_path.exists() and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Data file already exists: {_to_relative_repo_path(uploaded_path)}. Set overwrite=true to replace it.",
        )

    try:
        with uploaded_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
    finally:
        file.file.close()

    rows = _load_uploaded_json_array(uploaded_path)
    input_format = _detect_frontend_data_format(rows)
    alpaca_rows = _to_alpaca_rows(rows, input_format, instruction)

    from transformers import AutoTokenizer

    resolved_tokenizer_path = (
        _resolve_repo_path(tokenizer_path, "tokenizer_path")
        if tokenizer_path.strip()
        else (REPO_ROOT / "model" / "Llama-3.2-1B").resolve()
    )
    tokenizer = AutoTokenizer.from_pretrained(str(resolved_tokenizer_path))
    tokenizer.model_max_length = 10**12

    alpaca_path = alpaca_dir / f"{normalized_dataset_name}_{normalized_split_name}.json"
    bucket_paths = {
        bucket_name: alpaca_dir / f"{normalized_dataset_name}_{bucket_name}_{normalized_split_name}.json"
        for bucket_name, _, _ in bucket_specs
    }

    if alpaca_path.exists() and not overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"Alpaca file already exists: {_to_relative_repo_path(alpaca_path)}. Set overwrite=true to replace it.",
        )
    for path in bucket_paths.values():
        if path.exists() and not overwrite:
            raise HTTPException(
                status_code=409,
                detail=f"Bucketed file already exists: {_to_relative_repo_path(path)}. Set overwrite=true to replace it.",
            )

    enriched_rows: list[dict] = []
    bucketed_rows: dict[str, list[dict]] = {bucket_name: [] for bucket_name in bucket_paths}
    bucket_counts: dict[str, int] = {bucket_name: 0 for bucket_name in bucket_paths}

    for row in alpaca_rows:
        token_length = len(tokenizer.tokenize(row.get("input", ""))) + tokenizer.num_special_tokens_to_add(pair=False)
        enriched = dict(row)
        enriched["token_length"] = token_length
        bucket_name = _find_bucket(token_length, bucket_specs)
        enriched_rows.append(enriched)
        bucketed_rows[bucket_name].append(enriched)
        bucket_counts[bucket_name] += 1

    alpaca_path.write_text(json.dumps(enriched_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for bucket_name, path in bucket_paths.items():
        path.write_text(json.dumps(bucketed_rows[bucket_name], ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = alpaca_dir / f"{normalized_dataset_name}_{normalized_split_name}_rebucket_summary.json"
    summary_payload = {
        "dataset_name": normalized_dataset_name,
        "split_name": normalized_split_name,
        "input_format": input_format,
        "uploaded_path": _to_relative_repo_path(uploaded_path),
        "alpaca_path": _to_relative_repo_path(alpaca_path),
        "tokenizer_path": _to_relative_repo_path(resolved_tokenizer_path)
        if str(resolved_tokenizer_path).startswith(str(REPO_ROOT.resolve()))
        else str(resolved_tokenizer_path),
        "bucket_boundaries": boundaries,
        "total_samples": len(enriched_rows),
        "bucket_counts": bucket_counts,
    }
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return FrontendDataIngestResponse(
        dataset_name=normalized_dataset_name,
        split_name=normalized_split_name,
        input_format=input_format,  # type: ignore[arg-type]
        uploaded_path=_to_relative_repo_path(uploaded_path),
        alpaca_path=_to_relative_repo_path(alpaca_path),
        summary_path=_to_relative_repo_path(summary_path),
        bucket_files=[_to_relative_repo_path(path) for path in bucket_paths.values()],
        total_samples=len(enriched_rows),
        bucket_counts=bucket_counts,
    )


@app.get("/api/outputs", response_model=OutputListResponse)
def list_outputs(relative_path: str = Query(default="")) -> OutputListResponse:
    target = _resolve_outputs_path(relative_path)
    _assert_exists(target, "outputs path")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Target path is not a directory.")
    return _list_directory(OUTPUTS_ROOT, target)


@app.get("/api/outputs/text", response_model=OutputTextResponse)
def read_output_text(
    relative_path: str = Query(...),
    max_chars: int = Query(default=200000, ge=1, le=1000000),
) -> OutputTextResponse:
    target = _resolve_outputs_path(relative_path)
    _assert_exists(target, "output file")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Target path is not a file.")

    content = _read_text_file(target, max_chars=max_chars)
    return OutputTextResponse(
        base_dir=str(OUTPUTS_ROOT),
        relative_path=str(target.relative_to(OUTPUTS_ROOT.resolve())),
        content=content,
    )


@app.get("/api/outputs/file")
def get_output_file(relative_path: str = Query(...)) -> FileResponse:
    target = _resolve_outputs_path(relative_path)
    _assert_exists(target, "output file")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Target path is not a file.")

    media_type, _ = mimetypes.guess_type(str(target))
    return FileResponse(target, media_type=media_type or "application/octet-stream", filename=target.name)


@app.get("/api/frontend/user-data", response_model=OutputListResponse)
def list_frontend_user_data(relative_path: str = Query(default="")) -> OutputListResponse:
    if not relative_path.strip():
        return _list_user_root_directories(DATA_ROOT)

    target = _resolve_user_scoped_path(DATA_ROOT, relative_path, "data")
    _assert_exists(target, "data path")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Target path is not a directory.")
    return _list_directory(DATA_ROOT, target)


@app.get("/api/frontend/user-data/text", response_model=OutputTextResponse)
def read_frontend_user_data_text(
    relative_path: str = Query(...),
    max_chars: int = Query(default=200000, ge=1, le=1000000),
) -> OutputTextResponse:
    target = _resolve_user_scoped_path(DATA_ROOT, relative_path, "data")
    _assert_exists(target, "data file")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Target path is not a file.")

    content = _read_text_file(target, max_chars=max_chars)
    return OutputTextResponse(
        base_dir=str(DATA_ROOT),
        relative_path=str(target.relative_to(DATA_ROOT.resolve())),
        content=content,
    )


@app.post("/api/frontend/user-data/find-code", response_model=FrontendFindCodeResponse)
def frontend_find_code(payload: FrontendFindCodeRequest) -> FrontendFindCodeResponse:
    return _find_code_from_chunking_request(
        relative_path=payload.relative_path,
        sample_id=payload.sample_id,
        filename=payload.filename,
    )


@app.get("/api/frontend/user-outputs", response_model=OutputListResponse)
def list_frontend_user_outputs(relative_path: str = Query(default="")) -> OutputListResponse:
    if not relative_path.strip():
        return _list_user_root_directories(OUTPUTS_ROOT)

    target = _resolve_user_scoped_path(OUTPUTS_ROOT, relative_path, "outputs")
    _assert_exists(target, "outputs path")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Target path is not a directory.")
    return _list_directory(OUTPUTS_ROOT, target)


@app.get("/api/frontend/user-outputs/text", response_model=OutputTextResponse)
def read_frontend_user_outputs_text(
    relative_path: str = Query(...),
    max_chars: int = Query(default=200000, ge=1, le=1000000),
) -> OutputTextResponse:
    target = _resolve_user_scoped_path(OUTPUTS_ROOT, relative_path, "outputs")
    _assert_exists(target, "output file")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Target path is not a file.")

    content = _read_text_file(target, max_chars=max_chars)
    return OutputTextResponse(
        base_dir=str(OUTPUTS_ROOT),
        relative_path=str(target.relative_to(OUTPUTS_ROOT.resolve())),
        content=content,
    )


@app.get("/api/meta/options")
def get_options() -> dict[str, object]:
    datasets = _scan_data_directories()
    options = {
        "graph_models": sorted(GRAPH_MODELS),
        "llm_models": sorted(LLM_MODELS),
        "datasets": datasets,
        "regular_lengths": [],
        "imbalance_lengths": [],
        "imbalance_ratios": [],
    }

    regular_dirs = [REPO_ROOT / "data" / name / "alpaca" for name in datasets["regular"]]
    imbalance_dirs = [REPO_ROOT / "data" / name / "alpaca" for name in datasets["imbalance"]]

    if regular_dirs:
        options["regular_lengths"] = sorted(
            {
                item
                for directory in regular_dirs
                for item in _collect_suffixes(directory, _dataset_prefix_from_directory(directory), "_train.json")
                if item
            }
        )
    if imbalance_dirs:
        imbalance_lengths = set()
        imbalance_ratios = set()
        for directory in imbalance_dirs:
            lengths, ratios = _collect_imbalance_parts(directory)
            imbalance_lengths.update(lengths)
            imbalance_ratios.update(ratios)
        options["imbalance_lengths"] = sorted(imbalance_lengths)
        options["imbalance_ratios"] = sorted(imbalance_ratios)

    return options


@app.get("/api/meta/llm-test-options", response_model=LLMTestOptionsResponse)
def get_llm_test_options() -> LLMTestOptionsResponse:
    prompt_paths = sorted(LLM_TEST_PROMPT_ROOT.glob("*.txt")) if LLM_TEST_PROMPT_ROOT.exists() else []
    prompt_files = [str(path.relative_to(REPO_ROOT)) for path in prompt_paths]

    config_files = [
        str(path.relative_to(REPO_ROOT))
        for path in [LLM_TEST_ROOT / "exp.yaml", LLM_TEST_ROOT / ".env", LLM_TEST_ROOT / ".env.last_run"]
        if path.exists()
    ]
    env_files = [path for path in config_files if path.endswith(".env") or path.endswith(".env.last_run")]

    return LLMTestOptionsResponse(
        config_files=config_files,
        env_files=env_files,
        prompt_files=prompt_files,
        review_scripts=[
            "LLM_TEST/extract_positive_samples.py",
            "LLM_TEST/prepare_full_test_samples.py",
            "LLM_TEST/llm_api_judge.py",
            "LLM_TEST/recompute_metrics.py",
            "LLM_TEST/evaluate_llm_full_dataset.py",
            "LLM_TEST/run_review.sh",
            "LLM_TEST/ensemble_vote.py",
        ],
        ensemble_strategies=["any", "majority", "threshold", "weighted"],
        results_csv_options=_scan_glob_options(OUTPUTS_ROOT, "**/results.csv"),
        input_json_options=_scan_glob_options(LLM_TEST_INTERMEDIATE_ROOT, "**/positive_samples.jsonl"),
        full_input_json_options=_scan_glob_options(LLM_TEST_INTERMEDIATE_ROOT, "**/full_test_samples.jsonl"),
        llm_predictions_csv_options=_scan_glob_options(LLM_TEST_OUTPUT_ROOT, "**/llm_predictions.csv"),
        metrics_json_options=_scan_glob_options(LLM_TEST_OUTPUT_ROOT, "**/metrics.json"),
        full_metrics_json_options=_scan_glob_options(LLM_TEST_OUTPUT_ROOT, "**/full_llm_metrics.json"),
        llm_summary_options=_scan_glob_options(LLM_TEST_OUTPUT_ROOT, "**/llm_summary.json"),
        intermediate_dir_options=_scan_dir_options(LLM_TEST_INTERMEDIATE_ROOT, "positive_samples.jsonl"),
        output_dir_options=_scan_dir_options(LLM_TEST_OUTPUT_ROOT, "llm_predictions.csv"),
    )


@app.get("/api/meta/frontend-input-json-options", response_model=FrontendInputJsonOptionsResponse)
def get_frontend_input_json_options() -> FrontendInputJsonOptionsResponse:
    return FrontendInputJsonOptionsResponse(items=_scan_frontend_input_json_options())


@app.get("/api/llm-test/files", response_model=OutputListResponse)
def list_llm_test_files(
    root: str = Query(..., pattern="^(prompt|intermediate|output)$"),
    relative_path: str = Query(default=""),
) -> OutputListResponse:
    target = _resolve_llm_test_path(root, relative_path)
    _assert_exists(target, "LLM_TEST path")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Target path is not a directory.")
    return _list_directory(LLM_TEST_ALLOWED_ROOTS[root], target)


@app.get("/api/llm-test/files/text", response_model=OutputTextResponse)
def read_llm_test_text(
    root: str = Query(..., pattern="^(prompt|intermediate|output)$"),
    relative_path: str = Query(...),
    max_chars: int = Query(default=200000, ge=1, le=1000000),
) -> OutputTextResponse:
    target = _resolve_llm_test_path(root, relative_path)
    _assert_exists(target, "LLM_TEST file")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Target path is not a file.")

    base_root = LLM_TEST_ALLOWED_ROOTS[root]
    content = _read_text_file(target, max_chars=max_chars)
    return OutputTextResponse(
        base_dir=str(base_root),
        relative_path=str(target.relative_to(base_root.resolve())),
        content=content,
    )


@app.get("/api/llm-test/files/file")
def get_llm_test_file(
    root: str = Query(..., pattern="^(prompt|intermediate|output)$"),
    relative_path: str = Query(...),
) -> FileResponse:
    target = _resolve_llm_test_path(root, relative_path)
    _assert_exists(target, "LLM_TEST file")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Target path is not a file.")

    media_type, _ = mimetypes.guess_type(str(target))
    return FileResponse(target, media_type=media_type or "application/octet-stream", filename=target.name)


@app.post("/api/llm-test/prompts", response_model=PromptFileResponse)
def create_llm_test_prompt(payload: PromptCreateRequest) -> PromptFileResponse:
    prompt_file = _create_prompt_file(payload.prompt_name, payload.prompt_content, overwrite=payload.overwrite)
    return PromptFileResponse(
        path=_to_relative_repo_path(prompt_file),
        prompt_name=prompt_file.name,
        content=prompt_file.read_text(encoding="utf-8"),
    )


@app.post("/api/frontend/prompt-files", response_model=PromptFileResponse)
def create_frontend_prompt_file(payload: PromptCreateRequest) -> PromptFileResponse:
    return create_llm_test_prompt(payload)


@app.post("/api/jobs/classical", response_model=JobResponse)
def create_classical_job(payload: ClassicalJobRequest) -> JobResponse:
    if payload.model_name not in GRAPH_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported classical model.")
    script_name = "train.sh" if payload.action == "train" else "test.sh"
    output_dir = REPO_ROOT / "outputs" / payload.model_name / f"{payload.dataset_name}_{payload.length}"
    log_name = f"{payload.action}_{payload.model_name}_{payload.dataset_name}_{payload.length}.log"
    if payload.model_name == "Devign":
        log_path = output_dir / log_name
    else:
        log_path = output_dir / log_name
    return _launch_job(
        job_type="classical",
        script_name=script_name,
        args=[payload.dataset_name, payload.model_name, payload.length, payload.cuda],
        output_dir=output_dir,
        log_file=log_path,
        result_csv=output_dir / "results.csv",
        params=payload.model_dump(),
    )


@app.post("/api/frontend/code-inference", response_model=FrontendCodeInferenceResponse)
def frontend_code_inference(payload: FrontendCodeInferenceRequest) -> FrontendCodeInferenceResponse:
    checkpoint_dir = _resolve_checkpoint_dir(payload.checkpoint_dir)
    try:
        result = run_single_inference(
            model_name=payload.model_name,
            checkpoint_dir=checkpoint_dir,
            code=payload.code,
            block_size=payload.block_size,
            instruction=payload.instruction,
            sample_index=payload.sample_index,
            device_name=payload.device,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FrontendCodeInferenceResponse(**result)


@app.post("/api/frontend/code-inference-file", response_model=FrontendCodeInferenceFileResponse)
def frontend_code_inference_file(payload: FrontendCodeInferenceFileRequest) -> FrontendCodeInferenceFileResponse:
    checkpoint_dir = _resolve_checkpoint_dir(payload.checkpoint_dir)
    input_json = _resolve_repo_path(payload.input_json, "input_json")
    _assert_exists(input_json, "input_json")
    try:
        result = run_file_inference(
            model_name=payload.model_name,
            checkpoint_dir=checkpoint_dir,
            input_json_path=input_json,
            block_size=payload.block_size,
            device_name=payload.device,
            preview_limit=payload.preview_limit,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FrontendCodeInferenceFileResponse(**result)


@app.post("/api/frontend/code-chunking", response_model=FrontendCodeChunkResponse)
def frontend_code_chunking(payload: FrontendCodeChunkRequest) -> FrontendCodeChunkResponse:
    items: list[FrontendCodeChunkItem] = []

    if payload.items:
        items.extend(payload.items)

    if payload.code and payload.code.strip():
        items.append(
            FrontendCodeChunkItem(
                code=payload.code,
                language=payload.language,
                filename=payload.filename,
                sample_id=payload.sample_id,
            )
        )

    if not items:
        raise HTTPException(status_code=400, detail="Provide either code or items for chunking.")

    results: list[FrontendCodeChunkResult] = []
    flat_chunks: list[FrontendCodeChunkFlat] = []
    examples: list[FrontendCodeChunkExample] = []
    total_chunks = 0

    for item in items:
        try:
            chunking_result = run_code_chunking(
                code=item.code,
                language=item.language,
                max_chars=payload.max_chars,
                max_tokens=payload.max_tokens,
                fallback_lines=payload.fallback_lines,
                tokenizer_model_path=payload.tokenizer_model_path,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        chunk_count = int(chunking_result["chunk_count"])
        total_chunks += chunk_count
        results.append(
            FrontendCodeChunkResult(
                sample_id=item.sample_id,
                filename=item.filename,
                **chunking_result,
            )
        )

        for chunk in chunking_result["chunks"]:
            flat_chunks.append(
                FrontendCodeChunkFlat(
                    sample_id=item.sample_id,
                    filename=item.filename,
                    chunk_index=int(chunk["index"]),
                    text=str(chunk["text"]),
                    start_line=int(chunk["start_line"]),
                    end_line=int(chunk["end_line"]),
                    token_count=int(chunk["token_count"]),
                )
            )
            if len(examples) >= 5:
                break
            examples.append(
                FrontendCodeChunkExample(
                    sample_id=item.sample_id,
                    filename=item.filename,
                    chunk_index=int(chunk["index"]),
                    text=str(chunk["text"]),
                    start_line=int(chunk["start_line"]),
                    end_line=int(chunk["end_line"]),
                    token_count=int(chunk["token_count"]),
                )
            )

    response_payload = {
        "folder_name": f"user_{payload.folder_name.strip()}",
        "storage_dir": "",
        "total_inputs": len(items),
        "total_chunks": total_chunks,
        "chunks": [item.model_dump() for item in flat_chunks],
        "examples": [item.model_dump() for item in examples],
        "results": [item.model_dump() for item in results],
    }
    request_payload = payload.model_dump()

    try:
        storage_dir = save_chunking_payload(
            folder_name=payload.folder_name,
            max_tokens=payload.max_tokens,
            payload=response_payload,
            request_payload=request_payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    response_payload["storage_dir"] = str(storage_dir)

    return FrontendCodeChunkResponse(
        **response_payload,
    )


@app.post("/api/frontend/chunk-review", response_model=FrontendChunkReviewResponse)
def frontend_chunk_review(payload: FrontendChunkReviewRequest) -> FrontendChunkReviewResponse:
    prompt_file = None
    if payload.prompt_file or payload.prompt_name or payload.prompt_content:
        prompt_file = _resolve_or_create_prompt_file(
            prompt_file=payload.prompt_file,
            prompt_name=payload.prompt_name,
            prompt_content=payload.prompt_content,
            prompt_overwrite=payload.prompt_overwrite,
        )

    try:
        result = run_frontend_chunk_review(
            config_path=payload.config,
            env_file=payload.env_file,
            prompt_file=prompt_file,
            code=payload.code,
            chunks=[chunk.model_dump() for chunk in payload.chunks],
            language=payload.language,
            model=payload.model,
            api_base=payload.api_base,
            api_key=payload.api_key,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            timeout=payload.timeout,
            retries=payload.retries,
            sleep_seconds=payload.sleep_seconds,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FrontendChunkReviewResponse(**result)


@app.post("/api/jobs/classical-imbalance", response_model=JobResponse)
def create_imbalance_classical_job(payload: ImbalanceClassicalJobRequest) -> JobResponse:
    if payload.model_name not in GRAPH_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported classical model.")
    script_name = "train_imbalance.sh" if payload.action == "train" else "test_imbalance.sh"
    output_dir = REPO_ROOT / "outputs" / f"{payload.model_name}_imbalance" / f"{payload.dataset_name}_{payload.length}_{payload.pos_ratio}"
    log_path = output_dir / f"{payload.action}_{payload.model_name}_{payload.dataset_name}_{payload.length}_{payload.pos_ratio}.log"
    return _launch_job(
        job_type="classical_imbalance",
        script_name=script_name,
        args=[payload.dataset_name, payload.model_name, payload.length, payload.pos_ratio, payload.cuda],
        output_dir=output_dir,
        log_file=log_path,
        result_csv=output_dir / "results.csv",
        params=payload.model_dump(),
    )


@app.post("/api/jobs/llm", response_model=JobResponse)
def create_llm_job(payload: LLMJobRequest) -> JobResponse:
    if payload.model_name not in LLM_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported LLM model.")
    if payload.action == "finetune" and payload.batch_size is None:
        raise HTTPException(status_code=400, detail="batch_size is required for finetune.")
    script_name = "finetune.sh" if payload.action == "finetune" else "inference.sh"
    output_dir = REPO_ROOT / "outputs" / f"{payload.model_name}_lora" / f"{payload.dataset_name}_{payload.length}"
    log_prefix = "finetuning" if payload.action == "finetune" else "inference"
    log_path = output_dir / f"{log_prefix}_{payload.model_name}_lora_{payload.dataset_name}_{payload.length}.log"
    args = [payload.dataset_name, payload.model_name, payload.length]
    if payload.action == "finetune":
        args.append(payload.batch_size)
    args.append(payload.cuda)
    return _launch_job(
        job_type="llm",
        script_name=script_name,
        args=args,
        output_dir=output_dir,
        log_file=log_path,
        result_csv=output_dir / "results.csv",
        params=payload.model_dump(),
    )


@app.post("/api/jobs/llm-imbalance", response_model=JobResponse)
def create_imbalance_llm_job(payload: ImbalanceLLMJobRequest) -> JobResponse:
    if payload.model_name not in LLM_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported LLM model.")
    script_name = "finetune_imbalance.sh" if payload.action == "finetune" else "inference_imbalance.sh"
    output_dir = REPO_ROOT / "outputs" / f"{payload.model_name}_lora_imbalance" / f"{payload.dataset_name}_{payload.pos_ratio}"
    log_prefix = "finetuning" if payload.action == "finetune" else "inference"
    log_path = output_dir / f"{log_prefix}_{payload.model_name}_lora_{payload.dataset_name}_{payload.pos_ratio}.log"
    return _launch_job(
        job_type="llm_imbalance",
        script_name=script_name,
        args=[payload.dataset_name, payload.model_name, payload.pos_ratio, payload.cuda],
        output_dir=output_dir,
        log_file=log_path,
        result_csv=output_dir / "results.csv",
        params=payload.model_dump(),
    )


@app.post("/api/jobs/ablation", response_model=JobResponse)
def create_ablation_job(payload: AblationJobRequest) -> JobResponse:
    if payload.model_name not in LLM_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported LLM model.")
    script_name = "finetune_ablation.sh" if payload.action == "finetune" else "inference_ablation.sh"
    suffix = f"{payload.dataset_name}_{payload.r}_{payload.alpha}"
    output_dir = REPO_ROOT / "outputs" / f"{payload.model_name}_lora_ablation" / suffix
    log_suffix = f"{payload.model_name}_lora_ablation_{payload.dataset_name}_{payload.r}_{payload.alpha}.log"
    log_name = f"finetuning_{log_suffix}" if payload.action == "finetune" else f"inference_{log_suffix}"
    return _launch_job(
        job_type="ablation",
        script_name=script_name,
        args=[payload.dataset_name, payload.model_name, payload.r, payload.alpha, payload.cuda],
        output_dir=output_dir,
        log_file=output_dir / log_name,
        result_csv=output_dir / "results.csv",
        params=payload.model_dump(),
    )


@app.post("/api/jobs/to-graph", response_model=JobResponse)
def create_to_graph_job(payload: ToGraphJobRequest) -> JobResponse:
    output_dir = REPO_ROOT / "data" / payload.dataset_name / "graph"
    return _launch_job(
        job_type="to_graph",
        script_name="to_graph.sh",
        args=[payload.dataset_name, payload.length],
        output_dir=output_dir,
        log_file=None,
        result_csv=None,
        params=payload.model_dump(),
    )


@app.post("/api/jobs/llm-test/extract", response_model=JobResponse)
def create_llm_test_extract_job(payload: LLMTestExtractJobRequest) -> JobResponse:
    results_csv = _resolve_repo_path(payload.results_csv, "results_csv")
    data_root = _resolve_repo_path(payload.data_root, "data_root")
    output_root = _resolve_repo_path(payload.output_root, "output_root")
    data_json = _resolve_repo_path(payload.data_json, "data_json", allow_empty=True) if payload.data_json else None
    output_subdir = payload.output_subdir or results_csv.parent.name
    output_dir = output_root / output_subdir
    log_file = output_dir / "extract_positive_samples.log"

    command = [
        "python3",
        str(LLM_TEST_ROOT / "extract_positive_samples.py"),
        "--config",
        str(_resolve_repo_path(payload.config, "config")),
        "--env_file",
        str(_resolve_repo_path(payload.env_file, "env_file")),
        "--results_csv",
        str(results_csv),
        "--data_root",
        str(data_root),
        "--output_root",
        str(output_root),
        "--output_subdir",
        output_subdir,
        "--prediction_value",
        str(payload.prediction_value),
    ]
    if data_json is not None:
        command.extend(["--data_json", str(data_json)])
    if payload.min_prob is not None:
        command.extend(["--min_prob", str(payload.min_prob)])
    if payload.max_prob is not None:
        command.extend(["--max_prob", str(payload.max_prob)])

    return _launch_command_job(
        job_type="llm_test_extract",
        script_name="extract_positive_samples.py",
        command=command,
        output_dir=output_dir,
        log_file=log_file,
        result_csv=output_dir / "positive_samples.jsonl",
        params=payload.model_dump(),
    )


@app.post("/api/jobs/llm-test/full/prepare", response_model=JobResponse)
def create_llm_test_full_prepare_job(payload: LLMTestFullPrepareJobRequest) -> JobResponse:
    data_json = _resolve_full_prepare_data_json(payload.data_json)
    output_root = _resolve_repo_path(payload.output_root, "output_root")
    results_csv = _resolve_repo_path(payload.results_csv, "results_csv") if payload.results_csv else None
    output_subdir = payload.output_subdir or f"fulltest_{data_json.stem}"
    output_dir = output_root / output_subdir
    log_file = output_dir / "prepare_full_test_samples.log"

    command = [
        "python3",
        str(LLM_TEST_ROOT / "prepare_full_test_samples.py"),
        "--data_json",
        str(data_json),
        "--output_root",
        str(output_root),
        "--output_subdir",
        output_subdir,
    ]
    if results_csv is not None:
        command.extend(["--results_csv", str(results_csv)])
    if payload.limit is not None:
        command.extend(["--limit", str(payload.limit)])

    return _launch_command_job(
        job_type="llm_test_full_prepare",
        script_name="prepare_full_test_samples.py",
        command=command,
        output_dir=output_dir,
        log_file=log_file,
        result_csv=output_dir / "full_test_samples.jsonl",
        params=payload.model_dump(exclude_none=True),
    )


@app.post("/api/jobs/llm-test/judge", response_model=JobResponse)
def create_llm_test_judge_job(payload: LLMTestJudgeJobRequest) -> JobResponse:
    input_json = _resolve_repo_path(payload.input_json, "input_json")
    prompt_file = _resolve_or_create_prompt_file(
        prompt_file=payload.prompt_file,
        prompt_name=payload.prompt_name,
        prompt_content=payload.prompt_content,
        prompt_overwrite=payload.prompt_overwrite,
    )
    output_root = _resolve_repo_path(payload.output_root, "output_root")
    output_name = payload.output_name or input_json.parent.name
    output_dir = output_root / output_name
    log_file = output_dir / "llm_api_judge.log"

    command = [
        "python3",
        str(LLM_TEST_ROOT / "llm_api_judge.py"),
        "--config",
        str(_resolve_repo_path(payload.config, "config")),
        "--env_file",
        str(_resolve_repo_path(payload.env_file, "env_file")),
        "--input_json",
        str(input_json),
        "--prompt_file",
        str(prompt_file),
        "--output_root",
        str(output_root),
        "--workers",
        str(payload.workers),
        "--temperature",
        str(payload.temperature),
        "--max_tokens",
        str(payload.max_tokens),
        "--timeout",
        str(payload.timeout),
        "--retries",
        str(payload.retries),
        "--sleep_seconds",
        str(payload.sleep_seconds),
    ]
    if payload.output_name:
        command.extend(["--output_name", payload.output_name])
    if payload.output_by_prompt_version:
        command.append("--output_by_prompt_version")
    if payload.limit is not None:
        command.extend(["--limit", str(payload.limit)])
    if payload.model:
        command.extend(["--model", payload.model])
    if payload.api_base:
        command.extend(["--api_base", payload.api_base])
    if payload.api_key:
        command.extend(["--api_key", payload.api_key])
    if payload.fail_fast_on_error:
        command.append("--fail_fast_on_error")

    return _launch_command_job(
        job_type="llm_test_judge",
        script_name="llm_api_judge.py",
        command=command,
        output_dir=output_dir,
        log_file=log_file,
        result_csv=output_dir / "llm_predictions.csv",
        params=payload.model_dump(exclude={"api_key"}, exclude_none=True),
    )


@app.post("/api/jobs/llm-test/metrics", response_model=JobResponse)
def create_llm_test_metrics_job(payload: LLMTestMetricsJobRequest) -> JobResponse:
    results_csv = _resolve_repo_path(payload.results_csv, "results_csv")
    llm_predictions_csv = _resolve_repo_path(payload.llm_predictions_csv, "llm_predictions_csv")
    output_dir = (
        _resolve_repo_path(payload.output_dir, "output_dir")
        if payload.output_dir
        else llm_predictions_csv.parent
    )
    log_file = output_dir / "recompute_metrics.log"

    command = [
        "python3",
        str(LLM_TEST_ROOT / "recompute_metrics.py"),
        "--config",
        str(_resolve_repo_path(payload.config, "config")),
        "--env_file",
        str(_resolve_repo_path(payload.env_file, "env_file")),
        "--results_csv",
        str(results_csv),
        "--llm_predictions_csv",
        str(llm_predictions_csv),
        "--output_dir",
        str(output_dir),
    ]

    return _launch_command_job(
        job_type="llm_test_metrics",
        script_name="recompute_metrics.py",
        command=command,
        output_dir=output_dir,
        log_file=log_file,
        result_csv=output_dir / "metrics.json",
        params=payload.model_dump(exclude_none=True),
    )


@app.post("/api/jobs/llm-test/full/metrics", response_model=JobResponse)
def create_llm_test_full_metrics_job(payload: LLMTestFullMetricsJobRequest) -> JobResponse:
    llm_predictions_csv = _resolve_repo_path(payload.llm_predictions_csv, "llm_predictions_csv")
    output_dir = (
        _resolve_repo_path(payload.output_dir, "output_dir")
        if payload.output_dir
        else llm_predictions_csv.parent
    )
    log_file = output_dir / "evaluate_llm_full_dataset.log"

    command = [
        "python3",
        str(LLM_TEST_ROOT / "evaluate_llm_full_dataset.py"),
        "--env_file",
        str(_resolve_repo_path(payload.env_file, "env_file")),
        "--llm_predictions_csv",
        str(llm_predictions_csv),
        "--output_dir",
        str(output_dir),
    ]

    return _launch_command_job(
        job_type="llm_test_full_metrics",
        script_name="evaluate_llm_full_dataset.py",
        command=command,
        output_dir=output_dir,
        log_file=log_file,
        result_csv=output_dir / "full_llm_metrics.json",
        params=payload.model_dump(exclude_none=True),
    )


@app.post("/api/jobs/llm-test/review", response_model=JobResponse)
def create_llm_test_review_job(payload: LLMTestReviewJobRequest) -> JobResponse:
    config_path = _resolve_repo_path(payload.config, "config")
    env_path = _resolve_repo_path(payload.env, "env")
    output_root = _resolve_repo_path(payload.output_root, "output_root") if payload.output_root else None
    prompt_file = None
    if payload.prompt_file or payload.prompt_name or payload.prompt_content:
        prompt_file = _resolve_or_create_prompt_file(
            prompt_file=payload.prompt_file,
            prompt_name=payload.prompt_name,
            prompt_content=payload.prompt_content,
            prompt_overwrite=payload.prompt_overwrite,
        )

    command = [
        "bash",
        str(LLM_TEST_ROOT / "run_review.sh"),
        "--result-model",
        payload.result_model,
        "--dataset",
        payload.dataset,
        "--prob-range",
        payload.prob_range,
        "--workers",
        str(payload.workers),
        "--config",
        str(config_path),
        "--env",
        str(env_path),
    ]
    if payload.limit is not None:
        command.extend(["--limit", str(payload.limit)])
    if payload.data_json:
        command.extend(["--data-json", str(_resolve_repo_path(payload.data_json, "data_json"))])
    if prompt_file is not None:
        command.extend(["--prompt-file", str(prompt_file)])
    if output_root is not None:
        command.extend(["--output-root", str(output_root)])

    review_root = output_root if output_root is not None else LLM_TEST_OUTPUT_ROOT
    output_dir = review_root / f"{payload.dataset}_{payload.result_model}"
    log_file = output_dir / "run_review.log"

    return _launch_command_job(
        job_type="llm_test_review",
        script_name="run_review.sh",
        command=command,
        output_dir=output_dir,
        log_file=log_file,
        result_csv=None,
        params=payload.model_dump(exclude_none=True),
    )


@app.post("/api/jobs/llm-test/ensemble", response_model=JobResponse)
def create_llm_test_ensemble_job(payload: LLMTestEnsembleJobRequest) -> JobResponse:
    output_dir = (
        _resolve_repo_path(payload.output_dir, "output_dir")
        if payload.output_dir
        else _resolve_repo_path(payload.inputs[0].path, "inputs[0].path").parent
    )
    log_file = output_dir / f"{payload.output_prefix}_ensemble.log"

    command = [
        "python3",
        str(LLM_TEST_ROOT / "ensemble_vote.py"),
        "--inputs",
    ]
    for item in payload.inputs:
        input_path = _resolve_repo_path(item.path, f"inputs[{item.name}]")
        command.append(f"{item.name}={input_path}")
    command.extend(["--strategy", payload.strategy, "--output_dir", str(output_dir), "--output_prefix", payload.output_prefix])
    if payload.threshold is not None:
        command.extend(["--threshold", str(payload.threshold)])
    if payload.intersection_only:
        command.append("--intersection_only")
    if payload.weights:
        command.append("--weights")
        for name, weight in payload.weights.items():
            command.append(f"{name}={weight}")

    return _launch_command_job(
        job_type="llm_test_ensemble",
        script_name="ensemble_vote.py",
        command=command,
        output_dir=output_dir,
        log_file=log_file,
        result_csv=output_dir / f"{payload.output_prefix}_metrics.json",
        params=payload.model_dump(exclude_none=True),
    )


@app.get("/api/jobs", response_model=list[JobResponse])
def list_jobs() -> list[JobResponse]:
    return [_job_response(record) for record in job_manager.list_jobs()]


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    record = job_manager.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(record)


@app.get("/api/jobs/{job_id}/log", response_model=JobLogResponse)
def get_job_log(job_id: str, lines: int = Query(default=200, ge=1, le=2000)) -> JobLogResponse:
    record = job_manager.get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    log_file = Path(record.log_file) if record.log_file else None
    content = _tail_text(log_file, lines) if log_file else ""
    return JobLogResponse(job_id=job_id, log_file=record.log_file, content=content)


@app.post("/api/jobs/{job_id}/stop", response_model=JobResponse)
def stop_job(job_id: str) -> JobResponse:
    record = job_manager.stop_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _job_response(record)
