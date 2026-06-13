#!/usr/bin/env python3
"""Reformat and resplit contrastive method-pair JSONL datasets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from build_cvefixes_astchunk import normalize_cwe, split_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reformat contrastive method-pair JSONL files into the agreed schema and deterministic splits."
    )
    parser.add_argument(
        "--input_paths",
        nargs="+",
        required=True,
        help="One or more input JSONL files. You can pass existing train/validate/test files together.",
    )
    parser.add_argument("--output_root", required=True, help="Output root directory.")
    parser.add_argument("--dataset_name", required=True, help="Output dataset name.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate", action="store_true", help="Keep only the first record per pair_id.")
    return parser.parse_args()


def ordered_method(method: dict) -> dict:
    return {
        "method_change_id": method.get("method_change_id"),
        "name": method.get("name"),
        "signature": method.get("signature"),
        "start_line": method.get("start_line"),
        "end_line": method.get("end_line"),
        "code": method.get("code"),
    }


def ordered_record(record: dict) -> dict:
    return {
        "pair_id": record.get("pair_id"),
        "pair_key": record.get("pair_key"),
        "cve_id": record.get("cve_id"),
        "cwe_id": normalize_cwe(record.get("cwe_id")),
        "commit_hash": record.get("commit_hash"),
        "file_change_id": record.get("file_change_id"),
        "filename": record.get("filename"),
        "programming_language": record.get("programming_language"),
        "change_type": record.get("change_type"),
        "num_lines_added": int(record.get("num_lines_added") or 0),
        "num_lines_deleted": int(record.get("num_lines_deleted") or 0),
        "changed_lines": int(record.get("changed_lines") or 0),
        "diff": record.get("diff"),
        "before": ordered_method(record.get("before") or {}),
        "after": ordered_method(record.get("after") or {}),
        "before_changed_ranges": record.get("before_changed_ranges") or [],
        "after_changed_ranges": record.get("after_changed_ranges") or [],
    }


class SplitWriter:
    def __init__(self, output_root: Path, dataset_name: str):
        self.output_root = output_root
        self.dataset_name = dataset_name
        self.counts = defaultdict(int)
        self.paths = {
            split: self.output_root / dataset_name / "pairs" / f"{dataset_name}_{split}.jsonl"
            for split in ["train", "validate", "test"]
        }
        self.fps = {}
        for split, path in self.paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fps[split] = path.open("w", encoding="utf-8")

    def append(self, split: str, record: dict) -> None:
        out = ordered_record(record)
        out["index"] = self.counts[split]
        self.fps[split].write(json.dumps(out, ensure_ascii=False) + "\n")
        self.counts[split] += 1

    def close(self) -> None:
        for fp in self.fps.values():
            fp.close()


def iter_input_records(paths: list[str]):
    for path_str in paths:
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    args = parse_args()
    writer = SplitWriter(Path(args.output_root), args.dataset_name)
    seen_pair_ids: set[str] = set()
    total = 0
    kept = 0
    try:
        for record in iter_input_records(args.input_paths):
            total += 1
            pair_id = record.get("pair_id")
            if not pair_id:
                continue
            if args.deduplicate:
                if pair_id in seen_pair_ids:
                    continue
                seen_pair_ids.add(pair_id)
            split = split_name(pair_id, args.seed)
            writer.append(split, record)
            kept += 1
    finally:
        writer.close()

    stats = {
        "input_records": total,
        "output_records": kept,
        "train_records": writer.counts["train"],
        "validate_records": writer.counts["validate"],
        "test_records": writer.counts["test"],
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Postprocessed pairs written to {Path(args.output_root) / args.dataset_name / 'pairs'}")


if __name__ == "__main__":
    main()
