#!/usr/bin/env python3
"""Export CVEfixes DB records into ASTChunk-filtered Alpaca datasets.

Workflow:
- read before/after method contents from method_change and language from file_change
- pair method versions by signature, with a name fallback
- compute changed line ranges from method-level diffs in memory
- chunk each side with ASTChunk when the language is supported
- fall back to fixed line windows for unsupported languages or parser failures
- keep only chunks that overlap changed lines
- export Alpaca-style train/validate/test JSON files for the full dataset and CWEs
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import gc
from functools import lru_cache


ROOT = Path(__file__).resolve().parent
ASTCHUNK_SRC = ROOT / "astchunk-main" / "src"
if ASTCHUNK_SRC.exists():
    sys.path.insert(0, str(ASTCHUNK_SRC))

ASTCHUNK_IMPORT_ERROR = None
try:
    from astchunk import ASTChunkBuilder
except Exception as exc:  # pragma: no cover - environment dependent
    ASTChunkBuilder = None
    ASTCHUNK_IMPORT_ERROR = exc

TRANSFORMERS_IMPORT_ERROR = None
try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover - environment dependent
    AutoTokenizer = None
    TRANSFORMERS_IMPORT_ERROR = exc


INSTRUCTION = "Detect whether the following code contains vulnerabilities."
SUPPORTED_AST_LANGUAGES = {"python", "java", "csharp", "typescript", "cpp", "javascript", "php"}
DEFAULT_DB = ROOT / "Data" / "CVEfixes.db"
DEFAULT_TOKENIZER_MODEL = ROOT.parent / "LLM4CVD-main" / "model" / "Llama-3.2-1B"


@dataclass
class ChunkRecord:
    text: str
    start_line: int
    end_line: int


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default=str(DEFAULT_DB))
    p.add_argument("--output_root", default="data")
    p.add_argument(
        "--cwe_subdir",
        default="by_cwe",
        help="Subdirectory under output_root for CWE-specific datasets. Use empty string to keep the old flat layout.",
    )
    p.add_argument("--max_chars", type=int, default=1800)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument(
        "--tokenizer_model_path",
        default=str(DEFAULT_TOKENIZER_MODEL),
        help="Tokenizer/model path used for chunk length counting.",
    )
    p.add_argument("--fallback_lines", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min_cwe_samples", type=int, default=10)
    p.add_argument("--only_cwe", default=None, help="Optional, e.g. CWE-20 or 20")
    p.add_argument(
        "--cwes",
        nargs="+",
        default=None,
        help="Optional list of CWE ids to keep, e.g. 77 78 CWE-79 cwe22, or comma-separated values.",
    )
    p.add_argument("--limit", type=int, default=None, help="Optional row limit for debugging.")
    p.add_argument(
        "--batch_file_changes",
        type=int,
        default=20,
        help="Number of file_change_id values to process per SQL batch.",
    )
    p.add_argument("--log_file", default=None, help="Optional log file path.")
    p.add_argument("--progress_every", type=int, default=1000, help="Log progress every N method rows.")
    p.add_argument("--count_total", action="store_true", help="Run an extra COUNT(*) query to show percentage progress.")
    return p.parse_args()


class Logger:
    def __init__(self, log_path: str | None):
        self.log_fp = None
        if log_path:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.log_fp = path.open("w", encoding="utf-8")

    def log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, file=sys.stderr, flush=True)
        if self.log_fp is not None:
            self.log_fp.write(line + "\n")
            self.log_fp.flush()

    def close(self) -> None:
        if self.log_fp is not None:
            self.log_fp.close()


class JsonlDatasetWriter:
    def __init__(self, output_root: Path, cwe_subdir: str, max_tokens: int, min_cwe_samples: int):
        self.output_root = output_root
        self.cwe_output_base = output_root / cwe_subdir if cwe_subdir else output_root
        self.max_tokens = max_tokens
        self.min_cwe_samples = min_cwe_samples
        self.all_files = {
            split: self.output_root / "cvefixes" / "alpaca" / f"cvefixes_0-{self.max_tokens}_{split}.jsonl"
            for split in ["train", "validate", "test"]
        }
        self.all_fps = {
            split: self._open_output_file(path)
            for split, path in self.all_files.items()
        }
        self.cwe_files: dict[str, dict[str, Path]] = {}
        self.cwe_fps: dict[str, dict[str, object]] = {}
        self.cwe_counts = defaultdict(int)
        self.all_counts = defaultdict(int)
        self.cwe_split_counts = defaultdict(lambda: defaultdict(int))

    def _open_output_file(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("w", encoding="utf-8")

    def append(self, dataset_name: str | None, split: str, record: dict) -> None:
        full_record = dict(record)
        full_record["index"] = self.all_counts[split]
        line = json.dumps(full_record, ensure_ascii=False) + "\n"
        self.all_fps[split].write(line)
        self.all_counts[split] += 1

        if dataset_name is None:
            return

        line = json.dumps(record, ensure_ascii=False) + "\n"
        if dataset_name not in self.cwe_files:
            self.cwe_files[dataset_name] = {
                sp: self.cwe_output_base / dataset_name / "alpaca" / f"{dataset_name}_0-{self.max_tokens}_{sp}.jsonl"
                for sp in ["train", "validate", "test"]
            }
            self.cwe_fps[dataset_name] = {
                sp: self._open_output_file(path)
                for sp, path in self.cwe_files[dataset_name].items()
            }
        cwe_record = dict(record)
        cwe_record["index"] = self.cwe_split_counts[dataset_name][split]
        self.cwe_fps[dataset_name][split].write(json.dumps(cwe_record, ensure_ascii=False) + "\n")
        self.cwe_split_counts[dataset_name][split] += 1
        self.cwe_counts[dataset_name] += 1

    def finalize(self) -> tuple[int, int]:
        for fp in self.all_fps.values():
            fp.close()
        for fps in self.cwe_fps.values():
            for fp in fps.values():
                fp.close()

        total_records = sum(self.all_counts.values())
        exported = 0
        for dataset_name, count in sorted(self.cwe_counts.items()):
            if count < self.min_cwe_samples:
                continue
            exported += 1

        return total_records, exported


def normalize_cwe(cwe_id):
    if cwe_id is None:
        return None

    raw = str(cwe_id).strip()
    low = raw.lower()

    if "noinfo" in low or "other" in low or "unknown" in low:
        return None

    raw = re.sub(r"^(?i:nvd-cwe-)", "", raw)
    raw = re.sub(r"^(?i:cwe-?)", "", raw)
    return raw


def parse_cwe_filters(values: list[str] | None, only_cwe: str | None) -> set[str] | None:
    tokens: list[str] = []

    if only_cwe:
        tokens.append(only_cwe)

    if values:
        for value in values:
            parts = [part.strip() for part in str(value).split(",")]
            tokens.extend(part for part in parts if part)

    normalized = {cwe for cwe in (normalize_cwe(token) for token in tokens) if cwe is not None}
    return normalized or None


def build_cwe_match_values(selected_cwes: set[str] | None) -> list[str]:
    if not selected_cwes:
        return []

    values: set[str] = set()
    for cwe in selected_cwes:
        normalized = str(cwe).strip()
        values.add(normalized)
        values.add(f"CWE-{normalized}")
        values.add(f"NVD-CWE-{normalized}")
    return sorted(values)


def is_before_change(value) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def split_name(pair_id, seed):
    h = hashlib.md5((str(seed) + "::" + pair_id).encode("utf-8")).hexdigest()
    v = int(h[:8], 16) % 100
    if v < 80:
        return "train"
    if v < 90:
        return "validate"
    return "test"


def approximate_token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


@lru_cache(maxsize=4)
def load_tokenizer(model_path: str):
    if AutoTokenizer is None:
        raise RuntimeError(f"transformers import failed: {TRANSFORMERS_IMPORT_ERROR}")

    tokenizer_path = Path(model_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer model path does not exist: {tokenizer_path}")

    return AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)


def token_count(text: str, tokenizer_model_path: str | None) -> int:
    if tokenizer_model_path:
        try:
            tokenizer = load_tokenizer(tokenizer_model_path)
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return approximate_token_count(text)


def normalize_language(language: str | None) -> str | None:
    if not language:
        return None

    normalized = str(language).strip().lower()
    mapping = {
        "c": "cpp",
        "c++": "cpp",
        "cpp": "cpp",
        "cc": "cpp",
        "cxx": "cpp",
        "java": "java",
        "c#": "csharp",
        "csharp": "csharp",
        "cs": "csharp",
        "python": "python",
        "py": "python",
        "typescript": "typescript",
        "ts": "typescript",
        "javascript": "javascript",
        "js": "javascript",
        "php": "php",
    }
    return mapping.get(normalized)


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


def overlaps_changed_lines(chunk: ChunkRecord, changed_ranges: list[tuple[int, int]]) -> bool:
    for start, end in changed_ranges:
        if chunk.start_line <= end and start <= chunk.end_line:
            return True
    return False


def split_oversized_chunk(chunk: ChunkRecord, max_tokens: int, tokenizer_model_path: str | None) -> list[ChunkRecord]:
    chunk_tokens = token_count(chunk.text, tokenizer_model_path)
    if chunk_tokens <= max_tokens:
        return [chunk]

    lines = chunk.text.splitlines()
    if not lines:
        return []

    estimated_parts = max(2, math.ceil(chunk_tokens / max_tokens))
    window = max(1, math.ceil(len(lines) / estimated_parts))

    parts: list[ChunkRecord] = []
    start_idx = 0
    while start_idx < len(lines):
        end_idx = min(len(lines), start_idx + window)
        current_lines = lines[start_idx:end_idx]
        record = ChunkRecord(
            text="\n".join(current_lines),
            start_line=chunk.start_line + start_idx,
            end_line=chunk.start_line + len(current_lines) - 1,
        )

        while token_count(record.text, tokenizer_model_path) > max_tokens and len(current_lines) > 1:
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
    if ASTChunkBuilder is None:
        raise RuntimeError(f"ASTChunk is unavailable: {ASTCHUNK_IMPORT_ERROR}")

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


def build_chunks(
    code: str,
    language: str | None,
    max_chars: int,
    max_tokens: int,
    fallback_lines: int,
    tokenizer_model_path: str | None,
) -> tuple[list[ChunkRecord], str]:
    normalized_language = normalize_language(language)

    if normalized_language in SUPPORTED_AST_LANGUAGES:
        try:
            raw_chunks = ast_chunks(code, normalized_language, max_chars)
            chunk_source = f"ast:{normalized_language}"
        except Exception:
            raw_chunks = fixed_line_chunks(code, fallback_lines)
            chunk_source = f"fallback:{normalized_language}"
    else:
        raw_chunks = fixed_line_chunks(code, fallback_lines)
        chunk_source = f"fallback:{normalized_language or (language or 'unknown')}"

    final_chunks: list[ChunkRecord] = []
    for chunk in raw_chunks:
        final_chunks.extend(split_oversized_chunk(chunk, max_tokens, tokenizer_model_path))
    return final_chunks, chunk_source


def build_pair_key(row: sqlite3.Row) -> str:
    signature = (row["signature"] or "").strip()
    if signature:
        return f"sig::{signature}"
    return f"name::{(row['name'] or '').strip()}"


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def build_filtered_cves_cte(cwe_match_values: list[str]) -> tuple[str, list[str]]:
    if not cwe_match_values:
        return "", []

    placeholders = ",".join("?" for _ in cwe_match_values)
    cte_sql = f"""
    WITH filtered_cves AS (
        SELECT DISTINCT
            cc.cve_id,
            cc.cwe_id
        FROM cwe_classification cc
        WHERE cc.cwe_id IN ({placeholders})
    )
    """
    return cte_sql, list(cwe_match_values)


def fetch_target_file_change_ids(
    conn: sqlite3.Connection,
    cwe_match_values: list[str],
    limit: int | None,
) -> list[str]:
    cte_sql, params = build_filtered_cves_cte(cwe_match_values)

    if cwe_match_values:
        sql = cte_sql + """
        SELECT DISTINCT f.file_change_id
        FROM filtered_cves fc
        JOIN fixes fx ON fx.cve_id = fc.cve_id
        JOIN commits c ON c.hash = fx.hash
        JOIN file_change f ON f.hash = c.hash
        JOIN method_change m ON m.file_change_id = f.file_change_id
        WHERE m.code IS NOT NULL
          AND TRIM(m.code) != ''
        ORDER BY f.file_change_id
        """
    else:
        sql = """
        SELECT DISTINCT f.file_change_id
        FROM file_change f
        JOIN method_change m ON m.file_change_id = f.file_change_id
        WHERE m.code IS NOT NULL
          AND TRIM(m.code) != ''
        ORDER BY f.file_change_id
        """

    if limit is not None:
        sql += f" LIMIT {int(limit)}"

    return [row[0] for row in conn.execute(sql, params)]


def fetch_method_rows_for_file_changes(
    conn: sqlite3.Connection,
    file_change_ids: list[str],
    cwe_match_values: list[str],
) -> sqlite3.Cursor:
    if not file_change_ids:
        return conn.execute("SELECT 1 WHERE 0")

    file_placeholders = ",".join("?" for _ in file_change_ids)
    cte_sql, cte_params = build_filtered_cves_cte(cwe_match_values)

    if cwe_match_values:
        sql = cte_sql + f"""
        SELECT
            fx.cve_id,
            fc.cwe_id,
            fx.hash AS commit_hash,
            f.file_change_id,
            f.filename,
            f.programming_language,
            m.method_change_id,
            m.name,
            m.signature,
            m.before_change,
            m.code
        FROM filtered_cves fc
        JOIN fixes fx ON fx.cve_id = fc.cve_id
        JOIN commits c ON c.hash = fx.hash
        JOIN file_change f ON f.hash = c.hash
        JOIN method_change m ON m.file_change_id = f.file_change_id
        WHERE m.code IS NOT NULL
          AND TRIM(m.code) != ''
          AND f.file_change_id IN ({file_placeholders})
        ORDER BY
            f.file_change_id,
            CASE
                WHEN m.signature IS NOT NULL AND TRIM(m.signature) != '' THEN 'sig::' || TRIM(m.signature)
                ELSE 'name::' || TRIM(COALESCE(m.name, ''))
            END,
            m.method_change_id
        """
        params = cte_params + file_change_ids
    else:
        sql = f"""
        SELECT
            fx.cve_id,
            cc.cwe_id,
            fx.hash AS commit_hash,
            f.file_change_id,
            f.filename,
            f.programming_language,
            m.method_change_id,
            m.name,
            m.signature,
            m.before_change,
            m.code
        FROM file_change f
        JOIN method_change m ON m.file_change_id = f.file_change_id
        JOIN commits c ON f.hash = c.hash
        JOIN fixes fx ON c.hash = fx.hash
        JOIN cve cv ON fx.cve_id = cv.cve_id
        LEFT JOIN cwe_classification cc ON cv.cve_id = cc.cve_id
        WHERE m.code IS NOT NULL
          AND TRIM(m.code) != ''
          AND f.file_change_id IN ({file_placeholders})
        ORDER BY
            f.file_change_id,
            CASE
                WHEN m.signature IS NOT NULL AND TRIM(m.signature) != '' THEN 'sig::' || TRIM(m.signature)
                ELSE 'name::' || TRIM(COALESCE(m.name, ''))
            END,
            m.method_change_id
        """
        params = file_change_ids

    return conn.execute(sql, params)


def process_group(
    rows: list[sqlite3.Row],
    args: argparse.Namespace,
    writer: JsonlDatasetWriter,
    stats: dict[str, int],
    chunk_source_counts: defaultdict[str, int],
) -> None:
    if not rows:
        return

    before_rows = [row for row in rows if is_before_change(row["before_change"]) is True]
    after_rows = [row for row in rows if is_before_change(row["before_change"]) is False]

    pair_count = min(len(before_rows), len(after_rows))
    stats["candidate_pairs"] += pair_count
    stats["unpaired_methods"] += abs(len(before_rows) - len(after_rows))

    for before_row, after_row in zip(before_rows[:pair_count], after_rows[:pair_count]):
        cwe_id = normalize_cwe(before_row["cwe_id"])
        before_code = before_row["code"]
        after_code = after_row["code"]
        language = before_row["programming_language"]

        before_changed, after_changed = compute_changed_ranges(before_code, after_code)
        if not before_changed and not after_changed:
            continue
        stats["pairs_with_diff"] += 1

        before_chunks, before_source = build_chunks(
            before_code,
            language=language,
            max_chars=args.max_chars,
            max_tokens=args.max_tokens,
            fallback_lines=args.fallback_lines,
            tokenizer_model_path=args.tokenizer_model_path,
        )
        after_chunks, after_source = build_chunks(
            after_code,
            language=language,
            max_chars=args.max_chars,
            max_tokens=args.max_tokens,
            fallback_lines=args.fallback_lines,
            tokenizer_model_path=args.tokenizer_model_path,
        )

        chunk_source_counts[before_source] += 1
        chunk_source_counts[after_source] += 1

        kept_before = [chunk for chunk in before_chunks if overlaps_changed_lines(chunk, before_changed)]
        kept_after = [chunk for chunk in after_chunks if overlaps_changed_lines(chunk, after_changed)]
        if not kept_before or not kept_after:
            continue

        stats["pairs_kept"] += 1
        stats["before_chunks_kept"] += len(kept_before)
        stats["after_chunks_kept"] += len(kept_after)

        pair_id = (
            f"{before_row['cve_id']}::{before_row['commit_hash']}::{before_row['file_change_id']}::"
            f"{before_row['filename']}::{build_pair_key(before_row)}"
        )
        split = split_name(pair_id, args.seed)
        dataset_name = f"cvefixes_cwe{cwe_id}" if cwe_id is not None else None

        for chunk in kept_before:
            writer.append(
                dataset_name,
                split,
                {
                    "instruction": INSTRUCTION,
                    "input": chunk.text,
                    "output": "1",
                },
            )
        for chunk in kept_after:
            writer.append(
                dataset_name,
                split,
                {
                    "instruction": INSTRUCTION,
                    "input": chunk.text,
                    "output": "0",
                },
            )


def main():
    args = parse_args()
    selected_cwes = parse_cwe_filters(args.cwes, args.only_cwe)
    cwe_match_values = build_cwe_match_values(selected_cwes)
    output_root = Path(args.output_root)
    logger = Logger(args.log_file)

    if ASTChunkBuilder is None:
        logger.log(f"ASTChunk import failed, using fallback line chunks only: {ASTCHUNK_IMPORT_ERROR}")
    if args.tokenizer_model_path:
        try:
            load_tokenizer(args.tokenizer_model_path)
            logger.log(f"using tokenizer-based length counting from {args.tokenizer_model_path}")
        except Exception as exc:
            logger.log(
                f"failed to load tokenizer from {args.tokenizer_model_path}, "
                f"falling back to approximate token counting: {exc}"
            )

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-20000")
    conn.execute("PRAGMA mmap_size=0")
    writer = JsonlDatasetWriter(output_root, args.cwe_subdir, args.max_tokens, args.min_cwe_samples)

    stats = {
        "methods_total": 0,
        "candidate_pairs": 0,
        "pairs_with_diff": 0,
        "pairs_kept": 0,
        "unpaired_methods": 0,
        "before_chunks_kept": 0,
        "after_chunks_kept": 0,
    }
    chunk_source_counts = defaultdict(int)
    processed_groups = 0
    started_at = time.time()

    logger.log(
        f"start db={args.db} output_root={output_root} selected_cwes={sorted(selected_cwes) if selected_cwes else 'ALL'} "
        f"limit={args.limit} batch_file_changes={args.batch_file_changes}"
    )
    total_method_rows = None
    if args.count_total:
        count_sql = """
        SELECT COUNT(*)
        FROM method_change m
        JOIN file_change f ON m.file_change_id = f.file_change_id
        JOIN commits c ON f.hash = c.hash
        JOIN fixes fx ON c.hash = fx.hash
        JOIN cve cv ON fx.cve_id = cv.cve_id
        LEFT JOIN cwe_classification cc ON cv.cve_id = cc.cve_id
        WHERE m.code IS NOT NULL
          AND TRIM(m.code) != ''
        """
        if cwe_match_values:
            placeholders = ",".join("?" for _ in cwe_match_values)
            count_sql += f"""
          AND cc.cwe_id IN ({placeholders})
            """
            count_params = list(cwe_match_values)
        else:
            count_params = []
        logger.log("count_total enabled, running COUNT(*) query")
        total_method_rows = conn.execute(count_sql, count_params).fetchone()[0]
        logger.log(f"count_total finished total_method_rows={total_method_rows}")

    logger.log("fetching target file_change_id list")
    target_file_change_ids = fetch_target_file_change_ids(conn, cwe_match_values, args.limit)
    logger.log(f"target file_change_ids={len(target_file_change_ids)}")

    file_change_batches = chunked(target_file_change_ids, max(1, args.batch_file_changes))
    for batch_idx, batch_file_change_ids in enumerate(file_change_batches, start=1):
        logger.log(
            f"batch {batch_idx}/{len(file_change_batches)} fetching method rows for "
            f"{len(batch_file_change_ids)} file_change_ids"
        )
        cursor = fetch_method_rows_for_file_changes(conn, batch_file_change_ids, cwe_match_values)

        current_group_key: tuple[str, str] | None = None
        current_rows: list[sqlite3.Row] = []
        batch_method_rows = 0
        for row in cursor:
            batch_method_rows += 1
            stats["methods_total"] += 1
            group_key = (row["file_change_id"], build_pair_key(row))
            if current_group_key is None:
                current_group_key = group_key
            if group_key != current_group_key:
                process_group(current_rows, args, writer, stats, chunk_source_counts)
                processed_groups += 1
                current_rows = []
                current_group_key = group_key
            current_rows.append(row)
            if args.progress_every > 0 and stats["methods_total"] % args.progress_every == 0:
                elapsed = max(time.time() - started_at, 1e-6)
                rate = stats["methods_total"] / elapsed
                if total_method_rows is not None and total_method_rows > 0:
                    percent = 100.0 * stats["methods_total"] / total_method_rows
                    logger.log(
                        f"progress methods={stats['methods_total']}/{total_method_rows} ({percent:.1f}%) "
                        f"groups={processed_groups} kept_pairs={stats['pairs_kept']} "
                        f"out_records_so_far={sum(writer.all_counts.values())} rate={rate:.1f} rows/s"
                    )
                else:
                    logger.log(
                        f"progress methods={stats['methods_total']} groups={processed_groups} "
                        f"kept_pairs={stats['pairs_kept']} out_records_so_far={sum(writer.all_counts.values())} "
                        f"rate={rate:.1f} rows/s"
                    )

        process_group(current_rows, args, writer, stats, chunk_source_counts)
        processed_groups += 1 if current_rows else 0
        current_rows.clear()
        cursor.close()
        del cursor
        gc.collect()
        try:
            conn.execute("PRAGMA shrink_memory")
        except sqlite3.DatabaseError:
            pass
        logger.log(
            f"batch {batch_idx}/{len(file_change_batches)} finished method_rows={batch_method_rows} "
            f"groups={processed_groups} kept_pairs={stats['pairs_kept']} out_records_so_far={sum(writer.all_counts.values())}"
        )

    conn.close()
    logger.log("db scan finished, finalizing jsonl outputs")
    output_records, exported = writer.finalize()

    stats["output_records"] = output_records
    stats["chunk_sources"] = dict(sorted(chunk_source_counts.items()))
    if selected_cwes is not None:
        stats["selected_cwes"] = sorted(selected_cwes, key=lambda item: (len(item), item))
    logger.log(f"finished groups={processed_groups} output_records={output_records} exported_cwe_datasets={exported}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Exported full dataset to {output_root / 'cvefixes' / 'alpaca'}")
    print(f"Exported CWE datasets to {writer.cwe_output_base}: {exported}")
    logger.close()


if __name__ == "__main__":
    main()
