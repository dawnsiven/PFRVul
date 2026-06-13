#!/usr/bin/env python3
"""Expand contrastive method pairs into chunk-labeled JSONL datasets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from build_cvefixes_astchunk import (
    ChunkRecord,
    DEFAULT_TOKENIZER_MODEL,
    INSTRUCTION,
    build_chunks,
    overlaps_changed_lines,
    split_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pair-level before/after records into chunk-level labeled datasets."
    )
    parser.add_argument(
        "--input_paths",
        nargs="+",
        required=True,
        help="One or more pair-level JSONL files.",
    )
    parser.add_argument("--output_root", required=True, help="Output root directory.")
    parser.add_argument("--dataset_name", required=True, help="Output dataset name.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_chars", type=int, default=1800)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--fallback_lines", type=int, default=80)
    parser.add_argument(
        "--tokenizer_model_path",
        default=str(DEFAULT_TOKENIZER_MODEL),
        help="Tokenizer/model path used for chunk length counting. Use empty string to disable.",
    )
    parser.add_argument(
        "--output_format",
        choices=["flat", "alpaca"],
        default="flat",
        help="flat: metadata + code + label, alpaca: instruction/input/output with metadata.",
    )
    parser.add_argument("--deduplicate", action="store_true", help="Keep only the first record per pair_id.")
    return parser.parse_args()


def iter_input_records(paths: list[str]):
    for path_str in paths:
        path = Path(path_str)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


class ChunkWriter:
    def __init__(self, output_root: Path, dataset_name: str):
        self.output_root = output_root
        self.dataset_name = dataset_name
        self.counts = defaultdict(int)
        self.paths = {
            split: self.output_root / dataset_name / "chunks" / f"{dataset_name}_{split}.jsonl"
            for split in ["train", "validate", "test"]
        }
        self.fps = {}
        for split, path in self.paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            self.fps[split] = path.open("w", encoding="utf-8")

    def append(self, split: str, record: dict) -> None:
        out = dict(record)
        out["index"] = self.counts[split]
        self.fps[split].write(json.dumps(out, ensure_ascii=False) + "\n")
        self.counts[split] += 1

    def close(self) -> None:
        for fp in self.fps.values():
            fp.close()


def normalize_chunk_positions(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    normalized: list[ChunkRecord] = []
    next_start_line = 1
    for chunk in chunks:
        line_count = max(1, len(chunk.text.splitlines()))
        normalized.append(
            ChunkRecord(
                text=chunk.text,
                start_line=next_start_line,
                end_line=next_start_line + line_count - 1,
            )
        )
        next_start_line += line_count
    return normalized


def build_chunk_records(
    pair_record: dict,
    *,
    side: str,
    method: dict,
    changed_ranges: list[list[int]] | list[tuple[int, int]],
    max_chars: int,
    max_tokens: int,
    fallback_lines: int,
    tokenizer_model_path: str | None,
    output_format: str,
) -> list[dict]:
    code = method.get("code") or ""
    language = pair_record.get("programming_language")
    chunks, chunk_source = build_chunks(
        code,
        language=language,
        max_chars=max_chars,
        max_tokens=max_tokens,
        fallback_lines=fallback_lines,
        tokenizer_model_path=tokenizer_model_path,
    )
    chunks = normalize_chunk_positions(chunks)

    normalized_ranges = [(int(start), int(end)) for start, end in changed_ranges]
    records: list[dict] = []
    for chunk_id, chunk in enumerate(chunks):
        if side == "before":
            label = "1" if overlaps_changed_lines(chunk, normalized_ranges) else "0"
        else:
            label = "0"

        base = {
            "pair_id": pair_record.get("pair_id"),
            "pair_key": pair_record.get("pair_key"),
            "cve_id": pair_record.get("cve_id"),
            "cwe_id": pair_record.get("cwe_id"),
            "commit_hash": pair_record.get("commit_hash"),
            "file_change_id": pair_record.get("file_change_id"),
            "filename": pair_record.get("filename"),
            "programming_language": pair_record.get("programming_language"),
            "change_type": pair_record.get("change_type"),
            "changed_lines": int(pair_record.get("changed_lines") or 0),
            "side": side,
            "method_change_id": method.get("method_change_id"),
            "method_name": method.get("name"),
            "method_signature": method.get("signature"),
            "chunk_id": chunk_id,
            "chunk_start_line": chunk.start_line,
            "chunk_end_line": chunk.end_line,
            "chunk_source": chunk_source,
            "label": label,
        }
        if output_format == "alpaca":
            base.update(
                {
                    "instruction": INSTRUCTION,
                    "input": chunk.text,
                    "output": label,
                }
            )
        else:
            base.update(
                {
                    "code": chunk.text,
                    "output": label,
                }
            )
        records.append(base)
    return records


def main() -> None:
    args = parse_args()
    writer = ChunkWriter(Path(args.output_root), args.dataset_name)
    seen_pair_ids: set[str] = set()
    stats = {
        "input_pairs": 0,
        "output_pairs": 0,
        "before_chunks": 0,
        "after_chunks": 0,
        "positive_chunks": 0,
        "negative_chunks": 0,
    }
    tokenizer_model_path = args.tokenizer_model_path or None

    try:
        for pair_record in iter_input_records(args.input_paths):
            stats["input_pairs"] += 1
            pair_id = pair_record.get("pair_id")
            if not pair_id:
                continue
            if args.deduplicate:
                if pair_id in seen_pair_ids:
                    continue
                seen_pair_ids.add(pair_id)

            split = split_name(pair_id, args.seed)
            before_records = build_chunk_records(
                pair_record,
                side="before",
                method=pair_record.get("before") or {},
                changed_ranges=pair_record.get("before_changed_ranges") or [],
                max_chars=args.max_chars,
                max_tokens=args.max_tokens,
                fallback_lines=args.fallback_lines,
                tokenizer_model_path=tokenizer_model_path,
                output_format=args.output_format,
            )
            after_records = build_chunk_records(
                pair_record,
                side="after",
                method=pair_record.get("after") or {},
                changed_ranges=pair_record.get("after_changed_ranges") or [],
                max_chars=args.max_chars,
                max_tokens=args.max_tokens,
                fallback_lines=args.fallback_lines,
                tokenizer_model_path=tokenizer_model_path,
                output_format=args.output_format,
            )
            for record in before_records + after_records:
                writer.append(split, record)
                if record["side"] == "before":
                    stats["before_chunks"] += 1
                else:
                    stats["after_chunks"] += 1
                if record["output"] == "1":
                    stats["positive_chunks"] += 1
                else:
                    stats["negative_chunks"] += 1
            stats["output_pairs"] += 1
    finally:
        writer.close()

    stats["train_records"] = writer.counts["train"]
    stats["validate_records"] = writer.counts["validate"]
    stats["test_records"] = writer.counts["test"]
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Chunk dataset written to {Path(args.output_root) / args.dataset_name / 'chunks'}")


if __name__ == "__main__":
    main()
