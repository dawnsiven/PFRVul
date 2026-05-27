from __future__ import annotations

import importlib.util
import csv
import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Literal

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from fastapi_backend.job_runner import REPO_ROOT


InferenceModelName = Literal["CodeBERT", "UniXcoder"]
TEMP_INFERENCE_ROOT = REPO_ROOT / "data" / "temp_inference"
OUTPUTS_ROOT = REPO_ROOT / "outputs"
DEFAULT_INSTRUCTION = "Detect whether the following code contains vulnerabilities."

_MODULE_CACHE: dict[str, ModuleType] = {}
_MODEL_CACHE: dict[tuple[str, str, str], "_LoadedModel"] = {}
_CACHE_LOCK = threading.Lock()


@dataclass
class _LoadedModel:
    model_name: InferenceModelName
    checkpoint_dir: Path
    checkpoint_file: Path
    tokenizer: RobertaTokenizer
    model: torch.nn.Module
    device: torch.device


def _load_module(cache_key: str, module_path: Path) -> ModuleType:
    cached = _MODULE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    spec = importlib.util.spec_from_file_location(cache_key, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MODULE_CACHE[cache_key] = module
    return module


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no GPU is available.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    raise RuntimeError(f"Unsupported device: {device_name}")


def _normalize_code(code: str) -> str:
    return " ".join(code.split())


def create_temp_input(
    *,
    code: str,
    instruction: str = DEFAULT_INSTRUCTION,
    sample_index: int = 0,
) -> tuple[Path, Path]:
    temp_dir = TEMP_INFERENCE_ROOT / uuid.uuid4().hex
    temp_dir.mkdir(parents=True, exist_ok=False)

    sample_payload = [
        {
            "instruction": instruction,
            "input": code,
            "index": sample_index,
        }
    ]
    input_json_path = temp_dir / "input.json"
    input_json_path.write_text(
        json.dumps(sample_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "model_input_format": "frontend_single_sample",
        "has_ground_truth_label": False,
        "sample_count": 1,
    }
    (temp_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return temp_dir, input_json_path


def _load_codebert(checkpoint_dir: Path, device: torch.device) -> _LoadedModel:
    module = _load_module("fastapi_backend_codebert_model", REPO_ROOT / "CodeBERT" / "model.py")
    wrapper_cls = getattr(module, "Model")

    config = RobertaConfig.from_pretrained("microsoft/codebert-base")
    config.num_labels = 1
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    encoder = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", config=config)
    args = SimpleNamespace(dropout_probability=0.0, device=device)
    model = wrapper_cls(encoder, config, args)

    checkpoint_file = checkpoint_dir / "checkpoint-best-f1" / "model.bin"
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return _LoadedModel(
        model_name="CodeBERT",
        checkpoint_dir=checkpoint_dir,
        checkpoint_file=checkpoint_file,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def _load_unixcoder(checkpoint_dir: Path, device: torch.device) -> _LoadedModel:
    module = _load_module("fastapi_backend_unixcoder_model", REPO_ROOT / "UniXcoder" / "model.py")
    wrapper_cls = getattr(module, "Model")

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
    config = RobertaConfig.from_pretrained("microsoft/unixcoder-base")
    config.num_labels = 2
    encoder = RobertaForSequenceClassification.from_pretrained("microsoft/unixcoder-base", config=config)
    args = SimpleNamespace()
    model = wrapper_cls(encoder, config, args)

    checkpoint_file = checkpoint_dir / "checkpoint-best-f1" / "model.bin"
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return _LoadedModel(
        model_name="UniXcoder",
        checkpoint_dir=checkpoint_dir,
        checkpoint_file=checkpoint_file,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )


def get_loaded_model(
    *,
    model_name: InferenceModelName,
    checkpoint_dir: Path,
    device_name: str,
) -> _LoadedModel:
    checkpoint_dir = checkpoint_dir.resolve()
    device = _resolve_device(device_name)
    cache_key = (model_name, str(checkpoint_dir), str(device))

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        if model_name == "CodeBERT":
            loaded = _load_codebert(checkpoint_dir, device)
        elif model_name == "UniXcoder":
            loaded = _load_unixcoder(checkpoint_dir, device)
        else:
            raise RuntimeError(f"Unsupported model for direct inference: {model_name}")

        _MODEL_CACHE[cache_key] = loaded
        return loaded


def run_single_inference(
    *,
    model_name: InferenceModelName,
    checkpoint_dir: Path,
    code: str,
    block_size: int = 512,
    instruction: str = DEFAULT_INSTRUCTION,
    sample_index: int = 0,
    device_name: str = "auto",
) -> dict[str, object]:
    temp_dir, input_json_path = create_temp_input(
        code=code,
        instruction=instruction,
        sample_index=sample_index,
    )
    loaded = get_loaded_model(
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        device_name=device_name,
    )

    prediction, vulnerability_probability, non_vulnerable_probability = _predict_code(
        loaded=loaded,
        model_name=model_name,
        code=code,
        block_size=block_size,
    )
    result = {
        "model_name": model_name,
        "checkpoint_dir": str(loaded.checkpoint_dir),
        "checkpoint_file": str(loaded.checkpoint_file),
        "device": str(loaded.device),
        "prediction": prediction,
        "is_vulnerable": bool(prediction == 1),
        "vulnerability_probability": vulnerability_probability,
        "non_vulnerable_probability": non_vulnerable_probability,
        "temp_dir": str(temp_dir),
        "input_json": str(input_json_path),
    }
    (temp_dir / "prediction.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def _predict_code(
    *,
    loaded: _LoadedModel,
    model_name: InferenceModelName,
    code: str,
    block_size: int,
) -> tuple[int, float, float]:
    normalized_code = _normalize_code(code)
    if model_name == "CodeBERT":
        tokens = loaded.tokenizer.tokenize(normalized_code)[: block_size - 2]
        source_tokens = [loaded.tokenizer.cls_token] + tokens + [loaded.tokenizer.sep_token]
        source_ids = loaded.tokenizer.convert_tokens_to_ids(source_tokens)
        source_ids += [loaded.tokenizer.pad_token_id] * (block_size - len(source_ids))

        input_tensor = torch.tensor([source_ids], dtype=torch.long, device=loaded.device)
        with torch.no_grad():
            vulnerability_probability = float(loaded.model(input_ids=input_tensor).squeeze().item())
        prediction = 1 if vulnerability_probability > 0.5 else 0
        non_vulnerable_probability = 1.0 - vulnerability_probability
    else:
        tokens = loaded.tokenizer.tokenize(normalized_code)[: block_size - 4]
        source_tokens = [
            loaded.tokenizer.cls_token,
            "<encoder_only>",
            loaded.tokenizer.sep_token,
            *tokens,
            loaded.tokenizer.sep_token,
        ]
        source_ids = loaded.tokenizer.convert_tokens_to_ids(source_tokens)
        source_ids += [loaded.tokenizer.pad_token_id] * (block_size - len(source_ids))

        input_tensor = torch.tensor([source_ids], dtype=torch.long, device=loaded.device)
        with torch.no_grad():
            probs = loaded.model(input_ids=input_tensor).squeeze(0).detach().cpu().tolist()
        non_vulnerable_probability = float(probs[0])
        vulnerability_probability = float(probs[1])
        prediction = int(vulnerability_probability >= non_vulnerable_probability)
    return prediction, vulnerability_probability, non_vulnerable_probability


def _load_inference_items(input_json_path: Path) -> list[dict[str, object]]:
    payload = json.loads(input_json_path.read_text(encoding="utf-8"))
    items: list[dict[str, object]] = []

    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        running_index = 0
        for result in payload["results"]:
            if not isinstance(result, dict):
                continue
            sample_id = result.get("sample_id")
            filename = result.get("filename")
            for chunk in result.get("chunks", []):
                if not isinstance(chunk, dict):
                    continue
                code = str(chunk.get("text") or "").strip()
                if not code:
                    continue
                items.append(
                    {
                        "index": running_index,
                        "sample_id": sample_id,
                        "filename": filename,
                        "chunk_index": chunk.get("index"),
                        "code": code,
                        "source_item": {
                            "index": running_index,
                            "sample_id": sample_id,
                            "filename": filename,
                            "chunk_index": chunk.get("index"),
                            **chunk,
                        },
                    }
                )
                running_index += 1
        return items

    if isinstance(payload, list):
        for running_index, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            code = str(item.get("input", item.get("code", item.get("text", ""))) or "").strip()
            if not code:
                continue
            items.append(
                {
                    "index": int(item.get("index", running_index)),
                    "sample_id": item.get("sample_id"),
                    "filename": item.get("filename"),
                    "chunk_index": item.get("chunk_index"),
                    "code": code,
                    "source_item": dict(item),
                }
            )
        return items

    raise ValueError("Unsupported input_json format for file inference.")


def run_file_inference(
    *,
    model_name: InferenceModelName,
    checkpoint_dir: Path,
    input_json_path: Path,
    block_size: int = 512,
    device_name: str = "auto",
    preview_limit: int = 20,
) -> dict[str, object]:
    loaded = get_loaded_model(
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        device_name=device_name,
    )
    inference_items = _load_inference_items(input_json_path)
    if not inference_items:
        raise ValueError(f"No code samples found in {input_json_path}")

    result_rows: list[dict[str, object]] = []
    for item in inference_items:
        prediction, vulnerability_probability, non_vulnerable_probability = _predict_code(
            loaded=loaded,
            model_name=model_name,
            code=str(item["code"]),
            block_size=block_size,
        )
        result_rows.append(
            {
                "index": int(item["index"]),
                "sample_id": item.get("sample_id"),
                "filename": item.get("filename"),
                "chunk_index": item.get("chunk_index"),
                "prediction": prediction,
                "is_vulnerable": bool(prediction == 1),
                "vulnerability_probability": vulnerability_probability,
                "non_vulnerable_probability": non_vulnerable_probability,
                "code": str(item["code"]),
            }
        )

    positive_rows = [row for row in result_rows if row["prediction"] == 1]
    simplified_positive_rows: list[dict[str, object]] = []
    for row, source in zip(result_rows, inference_items):
        if row["prediction"] != 1:
            continue
        source_item = source.get("source_item")
        merged_row: dict[str, object] = {}
        if isinstance(source_item, dict):
            merged_row.update(source_item)
        merged_row.update(
            {
                "index": int(row["index"]),
                "sample_id": row.get("sample_id"),
                "filename": row.get("filename"),
                "chunk_index": row.get("chunk_index"),
                "prediction": 1,
                "vulnerability_probability": float(row["vulnerability_probability"]),
                "code": str(row["code"]),
            }
        )
        simplified_positive_rows.append(merged_row)

    output_dir = OUTPUTS_ROOT / input_json_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    result_stem = f"{input_json_path.stem}_{model_name}_frontend_inference"
    result_json = output_dir / f"{result_stem}.json"
    result_csv = output_dir / f"{result_stem}.csv"
    summary = {
        "model_name": model_name,
        "checkpoint_dir": str(loaded.checkpoint_dir),
        "checkpoint_file": str(loaded.checkpoint_file),
        "device": str(loaded.device),
        "input_json": str(input_json_path),
        "result_json": str(result_json),
        "result_csv": str(result_csv),
        "total_samples": len(result_rows),
        "vulnerable_samples": len(simplified_positive_rows),
        "results": simplified_positive_rows,
    }
    result_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "index",
        "sample_id",
        "filename",
        "chunk_index",
        "prediction",
        "vulnerability_probability",
        "code",
    ]
    csv_rows = [{field: row.get(field) for field in fieldnames} for row in simplified_positive_rows]
    with result_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    return {
        "model_name": model_name,
        "checkpoint_dir": str(loaded.checkpoint_dir),
        "checkpoint_file": str(loaded.checkpoint_file),
        "device": str(loaded.device),
        "input_json": str(input_json_path),
        "result_json": str(result_json),
        "result_csv": str(result_csv),
        "total_samples": len(result_rows),
        "vulnerable_samples": len(simplified_positive_rows),
        "preview": simplified_positive_rows[:preview_limit],
    }
