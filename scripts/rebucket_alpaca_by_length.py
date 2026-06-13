#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebucket split alpaca JSON files by tokenizer length."
    )
    parser.add_argument("--dataset-name", required=True, help="Dataset name, e.g. bigvul_cwe20")
    parser.add_argument("--length", required=True, help="Length tag in the source filename, e.g. 1")
    parser.add_argument("--pos-ratio", required=True, help="Positive ratio tag, e.g. 1")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to the parent of this script directory.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path or HF model id. Defaults to model/Llama-3.2-1B under repo root.",
    )
    parser.add_argument(
        "--bucket-boundaries",
        nargs="*",
        type=int,
        default=[512, 1024],
        help="Bucket cut points. Default: 512 1024",
    )
    parser.add_argument(
        "--output-dirname",
        default="alpaca_length_rebucketed",
        help="Subdirectory under data/<dataset>_subsampled for rebucketed files.",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_bucket_names(boundaries):
    values = sorted(set(boundaries))
    bucket_names = []
    lower = 0
    for upper in values:
        bucket_names.append((f"{lower}-{upper}", lower, upper))
        lower = upper
    bucket_names.append((f"{lower}-*", lower, None))
    return bucket_names


def compute_length(tokenizer, text):
    tokens = tokenizer(text, return_tensors="pt")
    ids = tokens["input_ids"].squeeze()
    return len(ids) if ids.dim() > 0 else 0


def find_bucket(length, bucket_specs):
    for bucket_name, lower, upper in bucket_specs:
        if upper is None:
            if length >= lower:
                return bucket_name
        elif lower <= length < upper:
            return bucket_name
    raise ValueError(f"Unable to bucket sample with length {length}")


def main():
    args = parse_args()
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[1]
    )
    dataset_tag = f"{args.dataset_name}_{args.length}_{args.pos_ratio}"
    source_dir = repo_root / "data" / f"{args.dataset_name}_subsampled" / "alpaca"
    output_dir = repo_root / "data" / f"{args.dataset_name}_subsampled" / args.output_dirname

    tokenizer_path = (
        Path(args.tokenizer_path).resolve()
        if args.tokenizer_path
        else repo_root / "model" / "Llama-3.2-1B"
    )
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    bucket_specs = build_bucket_names(args.bucket_boundaries)

    split_map = {
        "train": source_dir / f"{dataset_tag}_train.json",
        "validate": source_dir / f"{dataset_tag}_validate.json",
        "test": source_dir / f"{dataset_tag}_test.json",
    }

    print(f"Using tokenizer: {tokenizer_path}")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")

    for split, path in split_map.items():
        rows = load_json(path)
        bucketed = {bucket_name: [] for bucket_name, _, _ in bucket_specs}
        max_item = None

        for row in rows:
            text = row.get("input", "")
            length = compute_length(tokenizer, text)
            enriched = dict(row)
            enriched["token_length"] = length
            bucket_name = find_bucket(length, bucket_specs)
            bucketed[bucket_name].append(enriched)

            if max_item is None or length > max_item[1]:
                max_item = (row.get("index"), length)

        print(f"\nSplit: {split}")
        print(f"  total: {len(rows)}")
        print(f"  max: {max_item}")
        for bucket_name, _, _ in bucket_specs:
            output_path = output_dir / f"{dataset_tag}_{bucket_name}_{split}.json"
            save_json(output_path, bucketed[bucket_name])
            print(f"  {bucket_name}: {len(bucketed[bucket_name])} -> {output_path}")


if __name__ == "__main__":
    main()
