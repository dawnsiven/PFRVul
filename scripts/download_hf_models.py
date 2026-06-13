#!/usr/bin/env python3
import argparse
import inspect
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_HF_ENDPOINT = "https://huggingface.co"
COMMUNITY_MIRRORS = {
    "china": "https://hf-mirror.com",
}


MODEL_SPECS = {
    "codebert": {
        "repo_id": "microsoft/codebert-base",
        "local_dir": "codebert-base",
        "description": "CodeBERT / ReGVD default backbone",
    },
    "graphcodebert": {
        "repo_id": "microsoft/graphcodebert-base",
        "local_dir": "graphcodebert-base",
        "description": "GraphCodeBERT default backbone",
    },
    "unixcoder": {
        "repo_id": "microsoft/unixcoder-base",
        "local_dir": "unixcoder-base",
        "description": "UniXcoder default backbone",
    },
    "llama2": {
        "repo_id": "meta-llama/Llama-2-7b-hf",
        "local_dir": "Llama-2-7b-hf",
        "description": "LLM finetuning / inference base model",
    },
    "codellama": {
        "repo_id": "codellama/CodeLlama-7b-hf",
        "local_dir": "CodeLlama-7b-hf",
        "description": "LLM finetuning / inference base model",
    },
    "llama3": {
        "repo_id": "meta-llama/Meta-Llama-3-8B",
        "local_dir": "Meta-Llama-3-8B",
        "description": "LLM finetuning / inference base model",
    },
    "llama3.1": {
        "repo_id": "meta-llama/Llama-3.1-8B",
        "local_dir": "Llama-3.1-8B",
        "description": "LLM finetuning / inference base model",
    },
    "llama3.2": {
        "repo_id": "meta-llama/Llama-3.2-1B",
        "local_dir": "Llama-3.2-1B",
        "description": "Optional model used by finetune.sh",
    },
}


DEFAULT_MODELS = [
    "codebert",
    "graphcodebert",
    "unixcoder",
    "llama2",
    "codellama",
    "llama3",
    "llama3.1",
    "llama3.2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Hugging Face models used by this repository into model/."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS.keys()) + ["all"],
        default=["all"],
        help="Models to download. Default: all",
    )
    parser.add_argument(
        "--output-dir",
        default="model",
        help="Directory to store downloaded models. Default: model",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the target folder already exists.",
    )
    parser.add_argument(
        "--mirror",
        choices=sorted(COMMUNITY_MIRRORS.keys()) + ["official"],
        default=None,
        help="Select a download endpoint shortcut. Use 'china' for the community mirror or 'official' for huggingface.co.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Custom Hugging Face endpoint. Overrides --mirror and defaults to HF_ENDPOINT if set.",
    )
    parser.add_argument(
        "--resume-download",
        action="store_true",
        help="Deprecated compatibility flag. Downloads now resume automatically when possible.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only print the supported model aliases and exit.",
    )
    return parser.parse_args()


def resolve_selected_models(selected_args: list[str]) -> list[str]:
    if "all" in selected_args:
        return DEFAULT_MODELS
    return selected_args


def ensure_not_empty_dir(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def resolve_hf_endpoint(args: argparse.Namespace) -> str:
    if args.hf_endpoint:
        return args.hf_endpoint.rstrip("/")
    if args.mirror == "official":
        return DEFAULT_HF_ENDPOINT
    if args.mirror:
        return COMMUNITY_MIRRORS[args.mirror]
    return (os.getenv("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT).rstrip("/")


def print_supported_models() -> None:
    print("Supported model aliases:")
    for alias, spec in MODEL_SPECS.items():
        print(f"  - {alias:<13} -> {spec['repo_id']} ({spec['description']})")


def main() -> int:
    args = parse_args()

    if args.list:
        print_supported_models()
        return 0

    selected_models = resolve_selected_models(args.models)
    output_dir = Path(args.output_dir).resolve()
    hf_endpoint = resolve_hf_endpoint(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Download directory: {output_dir}")
    print(f"Selected models: {', '.join(selected_models)}")
    print(f"Hugging Face endpoint: {hf_endpoint}")

    if hf_endpoint != DEFAULT_HF_ENDPOINT:
        print(
            "Note: You are using a community/custom Hugging Face endpoint. "
            "Availability for gated repositories still depends on your Hugging Face access token and repo permissions."
        )

    if any(name.startswith("llama") or name == "codellama" for name in selected_models):
        print(
            "Note: Llama / CodeLlama models may require approved access on Hugging Face. "
            "If needed, set HF_TOKEN before running this script."
        )

    failures: list[tuple[str, str]] = []

    for name in selected_models:
        spec = MODEL_SPECS[name]
        local_path = output_dir / spec["local_dir"]

        if ensure_not_empty_dir(local_path) and not args.force:
            print(f"[skip] {name}: {local_path} already exists")
            continue

        print(f"[download] {name}: {spec['repo_id']} -> {local_path}")
        try:
            if args.resume_download:
                print(
                    "[note] --resume-download is no longer needed; huggingface_hub resumes automatically when possible."
                )
            download_kwargs = {
                "repo_id": spec["repo_id"],
                "local_dir": str(local_path),
                "token": args.token,
                "force_download": args.force,
            }
            if "endpoint" in inspect.signature(snapshot_download).parameters:
                download_kwargs["endpoint"] = hf_endpoint

            snapshot_download(
                **download_kwargs,
            )
            print(f"[done] {name}")
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"[failed] {name}: {exc}", file=sys.stderr)

    if failures:
        print("\nFailed downloads:", file=sys.stderr)
        for name, error in failures:
            print(f"  - {name}: {error}", file=sys.stderr)
        return 1

    print("\nAll requested models are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
