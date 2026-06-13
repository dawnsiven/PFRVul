#!/usr/bin/env python3
import json
from pathlib import Path


INSTRUCTION = "Detect whether the following code contains vulnerabilities."
SPLIT_NAME_MAP = {
    "train": "train",
    "valid": "validate",
    "test": "test",
}


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "code" not in row or "label" not in row or "index" not in row:
                raise KeyError(
                    f"Missing one of required fields code/label/index in {path}:{line_no}"
                )
            rows.append(
                {
                    "instruction": INSTRUCTION,
                    "input": row["code"],
                    "output": str(row["label"]),
                    "index": row["index"],
                }
            )
    return rows


def main():
    repo_root = Path(__file__).resolve().parent.parent
    dataset_dir = repo_root / "data" / "mvuld"
    output_dir = dataset_dir / "alpaca"
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_name, target_name in SPLIT_NAME_MAP.items():
        source_path = dataset_dir / f"{source_name}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source file: {source_path}")

        output_path = output_dir / f"mvuld_0-512_{target_name}.json"
        rows = load_jsonl(source_path)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=4, ensure_ascii=False)
            handle.write("\n")
        print(f"Wrote {len(rows)} samples to {output_path}")


if __name__ == "__main__":
    main()
