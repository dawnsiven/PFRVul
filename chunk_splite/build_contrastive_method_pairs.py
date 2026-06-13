#!/usr/bin/env python3
"""Build contrastive before/after method-pair datasets from local-fix file changes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from build_cvefixes_astchunk import (
    DEFAULT_DB,
    Logger,
    build_pair_key,
    compute_changed_ranges,
    is_before_change,
    normalize_cwe,
    split_name,
)


DEFAULT_FILTER_CSV = Path("data/single_modify_file_change_lengths.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export contrastive before/after method pairs from filtered local-fix file_change_ids."
    )
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--filter_csv", default=str(DEFAULT_FILTER_CSV))
    parser.add_argument("--output_root", default="data/contrastive_local_fixes")
    parser.add_argument("--dataset_name", default="cvefixes_local_fix_pairs")
    parser.add_argument("--max_changed_lines", type=int, default=40)
    parser.add_argument(
        "--max_method_code_chars",
        type=int,
        default=100000,
        help="Skip method_change rows whose code length exceeds this many characters. Use 0 to disable.",
    )
    parser.add_argument(
        "--max_diff_chars",
        type=int,
        default=100000,
        help="Skip file_change rows whose diff length exceeds this many characters. Use 0 to disable.",
    )
    parser.add_argument("--limit_file_changes", type=int, default=None)
    parser.add_argument("--batch_file_changes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ensure_indexes", action="store_true")
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--log_file", default=None)
    return parser.parse_args()


def load_filtered_file_change_ids(
    csv_path: Path,
    max_changed_lines: int,
    limit_file_changes: int | None,
) -> list[str]:
    file_change_ids: list[str] = []
    seen: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            changed_lines = int(row["changed_lines"] or 0)
            if changed_lines > max_changed_lines:
                continue
            file_change_id = (row["file_change_id"] or "").strip()
            if not file_change_id or file_change_id in seen:
                continue
            seen.add(file_change_id)
            file_change_ids.append(file_change_id)
            if limit_file_changes is not None and len(file_change_ids) >= limit_file_changes:
                break
    return file_change_ids


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def ensure_performance_indexes(conn: sqlite3.Connection, logger: Logger | None = None) -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS idx_method_change_file_change_id ON method_change(file_change_id)",
        "CREATE INDEX IF NOT EXISTS idx_file_change_file_change_id ON file_change(file_change_id)",
        "CREATE INDEX IF NOT EXISTS idx_file_change_hash ON file_change(hash)",
        "CREATE INDEX IF NOT EXISTS idx_fixes_hash ON fixes(hash)",
        "CREATE INDEX IF NOT EXISTS idx_cwe_classification_cve_id ON cwe_classification(cve_id)",
    ]
    for sql in statements:
        if logger is not None:
            logger.log(f"ensuring index: {sql}")
        conn.execute(sql)
    conn.commit()


def fetch_rows_for_file_changes(
    conn: sqlite3.Connection,
    file_change_ids: list[str],
    max_method_code_chars: int,
    max_diff_chars: int,
) -> sqlite3.Cursor:
    if not file_change_ids:
        return conn.execute("SELECT 1 WHERE 0")

    placeholders = ",".join("?" for _ in file_change_ids)
    where_clauses = [
        "m.code IS NOT NULL",
        "TRIM(m.code) != ''",
        f"f.file_change_id IN ({placeholders})",
    ]
    params: list[object] = list(file_change_ids)
    if max_method_code_chars > 0:
        where_clauses.append("length(m.code) <= ?")
        params.append(int(max_method_code_chars))
    if max_diff_chars > 0:
        where_clauses.append("(f.diff IS NULL OR length(f.diff) <= ?)")
        params.append(int(max_diff_chars))
    sql = f"""
    SELECT
        fx.cve_id,
        cc.cwe_id,
        f.hash AS commit_hash,
        f.file_change_id,
        f.filename,
        f.programming_language,
        f.change_type,
        f.num_lines_added,
        f.num_lines_deleted,
        f.diff,
        m.method_change_id,
        m.name,
        m.signature,
        m.before_change,
        m.start_line,
        m.end_line,
        m.code
    FROM file_change f
    JOIN method_change m ON m.file_change_id = f.file_change_id
    LEFT JOIN fixes fx ON fx.hash = f.hash
    LEFT JOIN cwe_classification cc ON cc.cve_id = fx.cve_id
    WHERE {" AND ".join(where_clauses)}
    ORDER BY
        f.file_change_id,
        CASE
            WHEN m.signature IS NOT NULL AND TRIM(m.signature) != '' THEN 'sig::' || TRIM(m.signature)
            ELSE 'name::' || TRIM(COALESCE(m.name, ''))
        END,
        m.method_change_id
    """
    return conn.execute(sql, params)


class PairDatasetWriter:
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
        out = dict(record)
        out["index"] = self.counts[split]
        self.fps[split].write(json.dumps(out, ensure_ascii=False) + "\n")
        self.counts[split] += 1

    def close(self) -> None:
        for fp in self.fps.values():
            fp.close()


def row_to_method_info(row: sqlite3.Row) -> dict:
    return {
        "method_change_id": row["method_change_id"],
        "name": row["name"],
        "signature": row["signature"],
        "start_line": int(row["start_line"]) if row["start_line"] not in (None, "") else None,
        "end_line": int(row["end_line"]) if row["end_line"] not in (None, "") else None,
        "code": row["code"],
    }


def build_record(before_row: sqlite3.Row, after_row: sqlite3.Row, pair_key: str) -> dict | None:
    before_code = before_row["code"]
    after_code = after_row["code"]
    before_ranges, after_ranges = compute_changed_ranges(before_code, after_code)
    if not before_ranges and not after_ranges:
        return None

    changed_lines = int(before_row["num_lines_added"] or 0) + int(before_row["num_lines_deleted"] or 0)
    pair_id = (
        f"{before_row['cve_id']}::{before_row['commit_hash']}::{before_row['file_change_id']}::"
        f"{before_row['filename']}::{pair_key}"
    )
    return {
        "pair_id": pair_id,
        "pair_key": pair_key,
        "cve_id": before_row["cve_id"],
        "cwe_id": normalize_cwe(before_row["cwe_id"]),
        "commit_hash": before_row["commit_hash"],
        "file_change_id": before_row["file_change_id"],
        "filename": before_row["filename"],
        "programming_language": before_row["programming_language"],
        "change_type": before_row["change_type"],
        "num_lines_added": int(before_row["num_lines_added"] or 0),
        "num_lines_deleted": int(before_row["num_lines_deleted"] or 0),
        "changed_lines": changed_lines,
        "diff": before_row["diff"],
        "before": row_to_method_info(before_row),
        "after": row_to_method_info(after_row),
        "before_changed_ranges": before_ranges,
        "after_changed_ranges": after_ranges,
    }


def main() -> None:
    args = parse_args()
    logger = Logger(args.log_file)
    output_root = Path(args.output_root)
    filter_csv = Path(args.filter_csv)

    target_file_change_ids = load_filtered_file_change_ids(
        filter_csv,
        max_changed_lines=args.max_changed_lines,
        limit_file_changes=args.limit_file_changes,
    )
    logger.log(
        f"loaded filtered file_change_ids={len(target_file_change_ids)} from {filter_csv} "
        f"with max_changed_lines<={args.max_changed_lines}"
    )
    logger.log(
        f"length filters max_method_code_chars={args.max_method_code_chars} "
        f"max_diff_chars={args.max_diff_chars}"
    )

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-20000")
    conn.execute("PRAGMA mmap_size=0")

    if args.ensure_indexes:
        ensure_performance_indexes(conn, logger)

    writer = PairDatasetWriter(output_root, args.dataset_name)
    stats = {
        "file_change_ids_seen": 0,
        "method_rows_seen": 0,
        "pair_groups_seen": 0,
        "candidate_pairs": 0,
        "kept_pairs": 0,
        "unpaired_groups": 0,
    }
    started_at = time.time()

    try:
        for batch_file_change_ids in chunked(target_file_change_ids, max(1, args.batch_file_changes)):
            cursor = fetch_rows_for_file_changes(
                conn,
                batch_file_change_ids,
                max_method_code_chars=args.max_method_code_chars,
                max_diff_chars=args.max_diff_chars,
            )
            grouped_rows: dict[str, dict[str, list[sqlite3.Row]]] = defaultdict(lambda: defaultdict(list))
            for row in cursor:
                stats["method_rows_seen"] += 1
                grouped_rows[str(row["file_change_id"])][build_pair_key(row)].append(row)
            cursor.close()

            for file_change_id in batch_file_change_ids:
                stats["file_change_ids_seen"] += 1
                groups = grouped_rows.get(str(file_change_id), {})
                for pair_key, rows in groups.items():
                    stats["pair_groups_seen"] += 1
                    before_rows = [row for row in rows if is_before_change(row["before_change"]) is True]
                    after_rows = [row for row in rows if is_before_change(row["before_change"]) is False]
                    pair_count = min(len(before_rows), len(after_rows))
                    stats["candidate_pairs"] += pair_count
                    if pair_count == 0:
                        stats["unpaired_groups"] += 1
                        continue
                    for before_row, after_row in zip(before_rows[:pair_count], after_rows[:pair_count]):
                        record = build_record(before_row, after_row, pair_key)
                        if record is None:
                            continue
                        split = split_name(record["pair_id"], args.seed)
                        writer.append(split, record)
                        stats["kept_pairs"] += 1

                if args.progress_every > 0 and stats["file_change_ids_seen"] % args.progress_every == 0:
                    elapsed = max(time.time() - started_at, 1e-6)
                    rate = stats["file_change_ids_seen"] / elapsed
                    logger.log(
                        f"progress file_change_ids={stats['file_change_ids_seen']} method_rows={stats['method_rows_seen']} "
                        f"candidate_pairs={stats['candidate_pairs']} kept_pairs={stats['kept_pairs']} "
                        f"unpaired_groups={stats['unpaired_groups']} rate={rate:.1f} file_changes/s"
                    )
            grouped_rows.clear()
    finally:
        writer.close()
        conn.close()
        logger.close()

    stats["train_records"] = writer.counts["train"]
    stats["validate_records"] = writer.counts["validate"]
    stats["test_records"] = writer.counts["test"]
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Exported contrastive pairs to {output_root / args.dataset_name / 'pairs'}")


if __name__ == "__main__":
    main()
