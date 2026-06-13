#!/usr/bin/env python3
"""
Build a chunked variant of an Alpaca-style code dataset.

Workflow:
- treat adjacent records as one tentative pair
- use labels inside the pair to resolve before (1) / after (0)
- compute line-level diffs in memory
- chunk each side with ASTChunk when language is supported
- fall back to fixed line windows for unsupported languages
- keep only chunks that overlap changed lines

If --output-dir is omitted and --source-dir looks like:
    data/<dataset_name>/alpaca
the default output directory becomes:
    data/<dataset_name>_astchunk/alpaca
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from astchunk import ASTChunkBuilder


INSTRUCTION = "Detect whether the following code contains vulnerabilities."
SUPPORTED_AST_LANGUAGES = {"python", "java", "csharp", "typescript", "cpp", "javascript", "php"}


@dataclass
class ChunkRecord:
    text: str
    start_line: int
    end_line: int


@dataclass
class PairResult:
    before: dict
    after: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_name",
        nargs="?",
        help="Dataset name under data/, for example: cvefixes_cwe29",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory containing Alpaca JSON files, typically data/<dataset>/alpaca",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write processed JSON files into. If omitted, derive from source-dir.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1800,
        help="ASTChunk max_chunk_size in non-whitespace characters.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum approximate token count per output chunk.",
    )
    parser.add_argument(
        "--fallback-lines",
        type=int,
        default=80,
        help="Fallback line window size for unsupported languages.",
    )
    return parser.parse_args()


def derive_output_dir(source_dir: Path) -> Path:
    source_dir = source_dir.resolve()
    if source_dir.name != "alpaca":
        return source_dir.parent / f"{source_dir.name}_astchunk"

    dataset_dir = source_dir.parent
    dataset_name = dataset_dir.name
    return dataset_dir.parent / f"{dataset_name}_astchunk" / "alpaca"


def derive_source_dir(dataset_name: str) -> Path:
    return Path("data") / dataset_name / "alpaca"


def list_source_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.glob("*.json") if path.is_file())


def approximate_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def infer_language(code: str) -> str | None:
    snippet = code[:3000]

    if re.search(r"^\s*<\?php", snippet, re.MULTILINE):
        return "php"
    if re.search(r"^\s*def\s+\w+\s*\(|^\s*class\s+\w+.*:", snippet, re.MULTILINE):
        return "python"
    if re.search(r"\b(public|private|protected)\s+static\s+function\b", snippet):
        return "php"
    if re.search(r"(?m)^\s*function\s+\w+\s*\(", snippet) and "$" in snippet:
        return "php"
    if re.search(r"(?m)^\s*(public|private|protected)?\s*class\s+\w+", snippet) and (
        ";" in snippet or "{" in snippet
    ):
        if "namespace " in snippet and "using " in snippet:
            return "csharp"
        if re.search(r"\b(Console|List<|IEnumerable<|namespace)\b", snippet):
            return "csharp"
        if re.search(r"\b(import\s+java\.|public\s+(class|interface|enum)\b)", snippet):
            return "java"
    if re.search(r"\b(import\s+java\.|public\s+(class|interface|enum)\b)", snippet):
        return "java"
    if re.search(r"\b(function|const|let|var)\b", snippet) and (
        "=>" in snippet or re.search(r":\s*[A-Za-z_][A-Za-z0-9_<>\[\]\|]*", snippet)
    ):
        return "typescript"
    if re.search(r"#include\s*<", snippet) or re.search(r"\bstd::\w+", snippet):
        return "cpp"
    if re.search(r"\bnamespace\s+\w+", snippet) and "::" in snippet:
        return "cpp"
    return None


def compute_changed_ranges(before_code: str, after_code: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    before_lines = before_code.splitlines()
    after_lines = after_code.splitlines()

    before_ranges: list[tuple[int, int]] = []
    after_ranges: list[tuple[int, int]] = []

    matcher = difflib.SequenceMatcher(a=before_lines, b=after_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i1 != i2:
            before_ranges.append((i1 + 1, i2))
        if j1 != j2:
            after_ranges.append((j1 + 1, j2))

    return merge_ranges(before_ranges), merge_ranges(after_ranges)


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def overlaps_changed_lines(chunk: ChunkRecord, changed_ranges: Iterable[tuple[int, int]]) -> bool:
    for start, end in changed_ranges:
        if chunk.start_line <= end and start <= chunk.end_line:
            return True
    return False


def split_oversized_chunk(chunk: ChunkRecord, max_tokens: int) -> list[ChunkRecord]:
    if approximate_token_count(chunk.text) <= max_tokens:
        return [chunk]

    lines = chunk.text.splitlines()
    if not lines:
        return []

    estimated_parts = max(2, math.ceil(approximate_token_count(chunk.text) / max_tokens))
    window = max(1, math.ceil(len(lines) / estimated_parts))

    parts: list[ChunkRecord] = []
    start_idx = 0
    while start_idx < len(lines):
        end_idx = min(len(lines), start_idx + window)
        current_lines = lines[start_idx:end_idx]
        record = ChunkRecord(
            text="\n".join(current_lines),
            start_line=chunk.start_line + start_idx,
            end_line=chunk.start_line + end_idx - 1,
        )

        while approximate_token_count(record.text) > max_tokens and len(current_lines) > 1:
            current_lines = current_lines[:-1]
            record = ChunkRecord(
                text="\n".join(current_lines),
                start_line=record.start_line,
                end_line=record.start_line + len(current_lines) - 1,
            )

        if record.text.strip():
            parts.append(record)
        start_idx = max(end_idx, start_idx + len(current_lines))

    return parts


def fixed_line_chunks(code: str, lines_per_chunk: int) -> list[ChunkRecord]:
    lines = code.splitlines()
    chunks: list[ChunkRecord] = []
    for start in range(0, len(lines), lines_per_chunk):
        window = lines[start : start + lines_per_chunk]
        if not window:
            continue
        chunks.append(
            ChunkRecord(
                text="\n".join(window),
                start_line=start + 1,
                end_line=start + len(window),
            )
        )
    return chunks


def ast_chunks(code: str, language: str, max_chars: int) -> list[ChunkRecord]:
    builder = ASTChunkBuilder(
        max_chunk_size=max_chars,
        language=language,
        metadata_template="default",
    )
    chunks = builder.chunkify(code, chunk_expansion=False, chunk_overlap=0)
    records: list[ChunkRecord] = []
    for chunk in chunks:
        metadata = chunk["metadata"]
        records.append(
            ChunkRecord(
                text=chunk["content"],
                start_line=int(metadata["start_line_no"]),
                end_line=int(metadata["end_line_no"]),
            )
        )
    return records


def build_chunks(code: str, max_chars: int, max_tokens: int, fallback_lines: int) -> tuple[list[ChunkRecord], str]:
    language = infer_language(code)
    if language in SUPPORTED_AST_LANGUAGES:
        try:
            raw_chunks = ast_chunks(code, language, max_chars)
            chunk_source = f"ast:{language}"
        except Exception:
            raw_chunks = fixed_line_chunks(code, fallback_lines)
            chunk_source = f"fallback:{language}"
    else:
        raw_chunks = fixed_line_chunks(code, fallback_lines)
        chunk_source = f"fallback:{language or 'unknown'}"

    final_chunks: list[ChunkRecord] = []
    for chunk in raw_chunks:
        final_chunks.extend(split_oversized_chunk(chunk, max_tokens))
    return final_chunks, chunk_source


def resolve_pair(first: dict, second: dict) -> PairResult | None:
    first_label = str(first.get("output", "")).strip()
    second_label = str(second.get("output", "")).strip()

    if first_label == "1" and second_label == "0":
        return PairResult(before=first, after=second)
    if first_label == "0" and second_label == "1":
        return PairResult(before=second, after=first)
    return None


def process_records(records: list[dict], max_chars: int, max_tokens: int, fallback_lines: int) -> tuple[list[dict], dict]:
    output_records: list[dict] = []
    stats = {
        "input_records": len(records),
        "tentative_pairs": len(records) // 2,
        "valid_pairs": 0,
        "skipped_invalid_pairs": 0,
        "empty_diff_pairs": 0,
        "before_chunks_kept": 0,
        "after_chunks_kept": 0,
    }
    chunk_source_counts: dict[str, int] = {}

    next_index = 0
    for pair_pos in range(0, len(records) - 1, 2):
        first = records[pair_pos]
        second = records[pair_pos + 1]
        pair = resolve_pair(first, second)
        if pair is None:
            stats["skipped_invalid_pairs"] += 1
            continue

        before_code = pair.before["input"]
        after_code = pair.after["input"]
        before_changed, after_changed = compute_changed_ranges(before_code, after_code)
        if not before_changed and not after_changed:
            stats["empty_diff_pairs"] += 1
            continue

        before_chunks, before_chunk_source = build_chunks(
            before_code, max_chars=max_chars, max_tokens=max_tokens, fallback_lines=fallback_lines
        )
        after_chunks, after_chunk_source = build_chunks(
            after_code, max_chars=max_chars, max_tokens=max_tokens, fallback_lines=fallback_lines
        )
        chunk_source_counts[before_chunk_source] = chunk_source_counts.get(before_chunk_source, 0) + 1
        chunk_source_counts[after_chunk_source] = chunk_source_counts.get(after_chunk_source, 0) + 1

        kept_before = [chunk for chunk in before_chunks if overlaps_changed_lines(chunk, before_changed)]
        kept_after = [chunk for chunk in after_chunks if overlaps_changed_lines(chunk, after_changed)]
        if not kept_before or not kept_after:
            continue

        stats["valid_pairs"] += 1
        stats["before_chunks_kept"] += len(kept_before)
        stats["after_chunks_kept"] += len(kept_after)

        for chunk in kept_before:
            output_records.append(
                {
                    "instruction": pair.before.get("instruction", INSTRUCTION),
                    "input": chunk.text,
                    "output": "1",
                    "index": next_index,
                }
            )
            next_index += 1

        for chunk in kept_after:
            output_records.append(
                {
                    "instruction": pair.after.get("instruction", INSTRUCTION),
                    "input": chunk.text,
                    "output": "0",
                    "index": next_index,
                }
            )
            next_index += 1

    stats["output_records"] = len(output_records)
    stats["chunk_sources"] = chunk_source_counts
    return output_records, stats


def main() -> None:
    args = parse_args()
    if args.source_dir is not None:
        source_dir = args.source_dir
    elif args.dataset_name:
        source_dir = derive_source_dir(args.dataset_name)
    else:
        raise ValueError("Provide either a dataset name or --source-dir.")

    output_dir = args.output_dir or derive_output_dir(source_dir)

    source_files = list_source_files(source_dir)
    if not source_files:
        raise FileNotFoundError(f"No JSON files found in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[source-dir] {source_dir.resolve()}")
    print(f"[output-dir] {output_dir.resolve()}")

    for source_path in source_files:
        output_path = output_dir / source_path.name

        records = json.loads(source_path.read_text(encoding="utf-8"))
        output_records, stats = process_records(
            records,
            max_chars=args.max_chars,
            max_tokens=args.max_tokens,
            fallback_lines=args.fallback_lines,
        )

        output_path.write_text(
            json.dumps(output_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"[processed] {source_path} -> {output_path}")
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
