#!/usr/bin/env python3
"""Convert JSONL files to JSON array files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Input .jsonl file or directory.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .json file or directory. If omitted, use the input path with a .json suffix.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When input_path is a directory, convert .jsonl files recursively.",
    )
    return parser.parse_args()


def convert_jsonl_file(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with input_path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{input_path}:{line_no}: invalid JSONL line: {exc}") from exc

    with output_path.open("w", encoding="utf-8") as fout:
        json.dump(records, fout, ensure_ascii=False, indent=2)
        fout.write("\n")
    return len(records)


def iter_jsonl_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.jsonl" if recursive else "*.jsonl"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_arg = Path(args.output) if args.output else None

    if input_path.is_file():
        if output_arg is None:
            output_path = input_path.with_suffix(".json")
        elif output_arg.exists() and output_arg.is_dir():
            output_path = output_arg / input_path.with_suffix(".json").name
        else:
            output_path = output_arg
        count = convert_jsonl_file(input_path, output_path)
        print(f"Converted {input_path} -> {output_path} ({count} records)")
        return

    if not input_path.is_dir():
        raise SystemExit(f"Input path not found: {input_path}")

    output_dir = output_arg if output_arg is not None else input_path
    files = iter_jsonl_files(input_path, args.recursive)
    if not files:
        print(f"No .jsonl files found under {input_path}")
        return

    converted = 0
    total_records = 0
    for jsonl_file in files:
        relative = jsonl_file.relative_to(input_path)
        output_path = (output_dir / relative).with_suffix(".json")
        total_records += convert_jsonl_file(jsonl_file, output_path)
        converted += 1

    print(f"Converted {converted} files with {total_records} total records")


if __name__ == "__main__":
    main()
