import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config_utils import get_section, load_env_file, load_yaml_config, resolve_value
from llm_api_judge import build_output_name, build_user_prompt, ensure_text, parse_binary_label

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LLM.utils.model_utils import resolve_model_source


MODEL_ALIASES = {
    "llama2": "meta-llama/Llama-2-7b-hf",
    "codellama": "codellama/CodeLlama-7b-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "llama3.1": "meta-llama/Llama-3.1-8B",
    "llama3.2": "meta-llama/Llama-3.2-1B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run zero-shot local Hugging Face LLM judging on extracted positive samples."
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
    parser.add_argument(
        "--model",
        default=None,
        help="Local path, Hugging Face repo id, or alias such as llama3.1 / llama3.2.",
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
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum generated tokens.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device placement: auto, cuda, cpu, cuda:0, etc.",
    )
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype for loading and inference.",
    )
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="Optional attention implementation passed to from_pretrained, such as flash_attention_2.",
    )
    return parser.parse_args()


def load_json(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        return json.load(handle)


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


def normalize_model_name(model_name: str) -> str:
    return MODEL_ALIASES.get(model_name, model_name)


def resolve_dtype(dtype_name: str) -> Optional[torch.dtype]:
    if dtype_name == "auto":
        return None
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main() -> None:
    args = parse_args()
    load_env_file(args.env_file)
    config = load_yaml_config(args.config)
    common_cfg = get_section(config, "common")
    llm_cfg = get_section(config, "llm")

    input_json_value = resolve_value(args.input_json, llm_cfg, "input_json")
    prompt_file_value = resolve_value(args.prompt_file, llm_cfg, "prompt_file")
    model_name = normalize_model_name(resolve_value(args.model, llm_cfg, "model"))
    output_root = resolve_value(args.output_root, common_cfg, "output_root", "LLM_TEST/output")
    temperature = float(resolve_value(args.temperature, llm_cfg, "temperature", 0.0))
    max_new_tokens = int(resolve_value(args.max_new_tokens, llm_cfg, "max_tokens", 64))

    if not model_name:
        raise ValueError("A local model path, Hugging Face repo id, or alias must be provided via --model.")

    input_json = Path(input_json_value).resolve()
    prompt_file = Path(prompt_file_value).resolve()
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

    model_source, use_local_model = resolve_model_source(model_name)
    if use_local_model:
        print(f"Using local model from {model_source}")
    else:
        print(f"Local model not found under model/, loading {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_source, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    device = choose_device(args.device)
    torch_dtype = resolve_dtype(args.torch_dtype)
    model_kwargs = {
        "device_map": "auto" if device == "cuda" else None,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()
    input_device = next(model.parameters()).device

    judgment_rows: List[dict] = []
    csv_rows: List[dict] = []

    for sample in positive_samples:
        user_prompt = build_user_prompt(prompt_template, sample.get("input", ""))
        model_input = tokenizer(user_prompt, return_tensors="pt", truncation=True)
        model_input = {key: value.to(input_device) for key, value in model_input.items()}

        with torch.no_grad():
            generation = model.generate(
                **model_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_length = model_input["input_ids"].shape[1]
        generated_tokens = generation[0][prompt_length:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        llm_prediction, parse_source = parse_binary_label(response_text)
        parse_status = parse_source or "unparsed"

        judgment = {
            "dataset_id": dataset_id,
            "index": sample["index"],
            "label": sample["ground_truth"],
            "original_prediction": sample["original_prediction"],
            "llm_prediction": llm_prediction,
            "model": model_name,
            "model_source": model_source,
            "parse_status": parse_status,
            "prompt_file": str(prompt_file),
            "raw_response": response_text,
        }
        judgment_rows.append(judgment)
        csv_rows.append(
            {
                "Index": sample["index"],
                "Label": sample["ground_truth"],
                "OriginalPrediction": sample["original_prediction"],
                "LLMPrediction": "" if llm_prediction is None else llm_prediction,
                "LLMModel": model_name,
                "ParseStatus": parse_status,
                "RawResponse": response_text,
            }
        )

        print(
            f"index={sample['index']} label={sample['ground_truth']} "
            f"original={sample['original_prediction']} llm={llm_prediction} status={parse_status}"
        )

    write_jsonl(output_dir / "llm_judgments.jsonl", judgment_rows)
    write_csv(output_dir / "llm_predictions.csv", csv_rows)
    write_json(
        output_dir / "llm_summary.json",
        {
            "dataset_id": dataset_id,
            "output_name": output_name,
            "input_json": str(input_json),
            "prompt_file": str(prompt_file),
            "model": model_name,
            "model_source": model_source,
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
