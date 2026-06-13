#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebucket reviewer train/val/test JSON files by tokenizer length."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing train.json, val.json, and test.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to a sibling directory named <input>_length_rebucketed.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path or model id. Defaults to model/Llama-3.2-1B under repo root.",
    )
    parser.add_argument(
        "--bucket-boundaries",
        nargs="*",
        type=int,
        default=[512, 1024],
        help="Bucket cut points. Default: 512 1024",
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


def build_bucket_specs(boundaries):
    values = sorted(set(boundaries))
    specs = []
    lower = 0
    for upper in values:
        specs.append((f"{lower}-{upper}", lower, upper))
        lower = upper
    specs.append((f"{lower}-*", lower, None))
    return specs


def compute_length(tokenizer, text):
    tokens = tokenizer(text, return_tensors="pt")
    ids = tokens["input_ids"].squeeze()
    return len(ids) if ids.dim() > 0 else 0


def find_bucket(length, bucket_specs):
    for bucket_name, lower, upper in bucket_specs:
        if upper is None and length >= lower:
            return bucket_name
        if upper is not None and lower <= length < upper:
            return bucket_name
    raise ValueError(f"Unable to bucket sample length {length}")


def bucket_spec_to_dict(bucket_specs):
    items = []
    for bucket_name, lower, upper in bucket_specs:
        items.append(
            {
                "name": bucket_name,
                "lower_inclusive": lower,
                "upper_exclusive": upper,
            }
        )
    return items


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    tokenizer_path = (
        Path(args.tokenizer_path).resolve()
        if args.tokenizer_path
        else repo_root / "model" / "Llama-3.2-1B"
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir.parent / f"{input_dir.name}_length_rebucketed"
    )

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    bucket_specs = build_bucket_specs(args.bucket_boundaries)
    split_files = {
        "train": input_dir / "train.json",
        "val": input_dir / "val.json",
        "test": input_dir / "test.json",
    }

    print(f"Using tokenizer: {tokenizer_path}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    summary = {
        "tokenizer_path": str(tokenizer_path),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "bucket_boundaries": list(args.bucket_boundaries),
        "bucket_specs": bucket_spec_to_dict(bucket_specs),
        "splits": {},
    }

    for split_name, path in split_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")

        rows = load_json(path)
        bucketed = {bucket_name: [] for bucket_name, _, _ in bucket_specs}
        max_item = None

        for row in rows:
            length = compute_length(tokenizer, row.get("input", ""))
            enriched = dict(row)
            enriched["token_length"] = length
            bucket_name = find_bucket(length, bucket_specs)
            bucketed[bucket_name].append(enriched)
            if max_item is None or length > max_item[1]:
                max_item = (row.get("index"), length)

        print(f"\nSplit: {split_name}")
        print(f"  total: {len(rows)}")
        print(f"  max: {max_item}")
        split_summary = {
            "source_file": str(path),
            "total": len(rows),
            "max": {
                "index": None if max_item is None else max_item[0],
                "token_length": None if max_item is None else max_item[1],
            },
            "buckets": {},
        }
        for bucket_name, _, _ in bucket_specs:
            output_path = output_dir / f"{split_name}_{bucket_name}.json"
            save_json(output_path, bucketed[bucket_name])
            print(f"  {bucket_name}: {len(bucketed[bucket_name])} -> {output_path}")
            split_summary["buckets"][bucket_name] = {
                "count": len(bucketed[bucket_name]),
                "output_file": str(output_path),
            }
        summary["splits"][split_name] = split_summary

    summary_path = output_dir / "rebucket_summary.json"
    save_json(summary_path, summary)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
