#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_SEARCH_ROOTS = ["outputs", "outputs_1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively scan directories for merged_metrics.json files and "
            "export them as a single flat CSV table."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=DEFAULT_SEARCH_ROOTS,
        help=(
            "Directories to scan recursively. "
            f"Defaults to: {' '.join(DEFAULT_SEARCH_ROOTS)}"
        ),
    )
    parser.add_argument(
        "--output",
        default="analysis/merged_metrics_summary.csv",
        help="Path to the exported CSV file.",
    )
    parser.add_argument(
        "--pattern",
        default="merged_metrics.json",
        help="Filename pattern to match during recursive scanning.",
    )
    return parser.parse_args()


def json_scalar(value: object) -> object:
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, bool):
        return int(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def flatten_dict(data: Dict[str, object], prefix: str = "") -> Dict[str, object]:
    flat: Dict[str, object] = {}
    for key, value in data.items():
        new_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key))
        else:
            flat[new_key] = json_scalar(value)
    return flat


def find_metric_files(roots: Iterable[str], pattern: str) -> List[Tuple[Path, Path]]:
    matched: List[Tuple[Path, Path]] = []
    for root_text in roots:
        root = Path(root_text).resolve()
        if not root.exists():
            print(f"[warn] skip missing root: {root}")
            continue
        for path in sorted(root.rglob(pattern)):
            matched.append((root, path.resolve()))
    return matched


def build_metadata(scan_root: Path, metrics_path: Path) -> Dict[str, object]:
    parent_dir = metrics_path.parent
    try:
        relative_dir = parent_dir.relative_to(scan_root)
    except ValueError:
        relative_dir = parent_dir

    relative_dir_str = str(relative_dir)
    path_parts = list(relative_dir.parts)

    metadata: Dict[str, object] = {
        "scan_root": str(scan_root),
        "metrics_json": str(metrics_path),
        "relative_dir": relative_dir_str,
        "experiment_dir": parent_dir.name,
        "group_dir": path_parts[0] if path_parts else "",
        "path_depth": len(path_parts),
    }

    for index, part in enumerate(path_parts):
        metadata[f"path_part_{index}"] = part

    return metadata


def load_row(scan_root: Path, metrics_path: Path) -> Dict[str, object]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    row = build_metadata(scan_root, metrics_path)
    row.update(flatten_dict(payload))
    return row


def sort_fieldnames(fieldnames: Iterable[str]) -> List[str]:
    preferred_prefixes = [
        "scan_root",
        "metrics_json",
        "relative_dir",
        "experiment_dir",
        "group_dir",
        "path_depth",
        "path_part_",
        "merged_results_csv",
        "summary_",
        "original_metrics_",
        "final_metrics_",
        "reviewed_subset_original_metrics_",
        "reviewed_subset_final_metrics_",
    ]

    def field_key(name: str) -> Tuple[int, str]:
        for index, prefix in enumerate(preferred_prefixes):
            if name == prefix or name.startswith(prefix):
                return index, name
        return len(preferred_prefixes), name

    return sorted(fieldnames, key=field_key)


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sort_fieldnames({key for row in rows for key in row.keys()})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    matches = find_metric_files(args.roots, args.pattern)

    if not matches:
        raise SystemExit("No merged_metrics.json files were found under the given roots.")

    rows = [load_row(scan_root, metrics_path) for scan_root, metrics_path in matches]
    output_path = Path(args.output).resolve()
    write_csv(rows, output_path)

    print(f"Exported {len(rows)} rows to: {output_path}")


if __name__ == "__main__":
    main()
