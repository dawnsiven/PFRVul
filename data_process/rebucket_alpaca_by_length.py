#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebucket Alpaca-format JSON files by tokenizer length in place."
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Dataset directory containing the alpaca subdirectory.",
    )
    parser.add_argument(
        "--dataset-name",
        required=True,
        help="Dataset name used in output filenames, e.g. cvefixes_cwe77.",
    )
    parser.add_argument(
        "--input-prefix",
        required=True,
        help="Original filename prefix inside alpaca1, e.g. cvefixes_cwe77_0-512.",
    )
    parser.add_argument(
        "--source-dirname",
        default="alpaca",
        help="Original Alpaca directory name. Default: alpaca",
    )
    parser.add_argument(
        "--backup-dirname",
        default="alpaca1",
        help="Backup directory name created from the original Alpaca directory. Default: alpaca1",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="Tokenizer path or model id. Defaults to repo_root/model/Llama-3.2-1B.",
    )
    parser.add_argument(
        "--bucket-boundaries",
        nargs="*",
        type=int,
        default=[512, 1024],
        help="Bucket cut points. Default: 512 1024",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "validate", "test"],
        help="Splits to process. Default: train validate test",
    )
    return parser.parse_args()


def iter_json_array(path: Path, chunk_size: int = 1 << 20):
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    finished = False

    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            position = 0

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1

                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError(f"{path} is not a JSON array")
                    started = True
                    position += 1
                    continue

                while position < len(buffer) and buffer[position].isspace():
                    position += 1

                if position >= len(buffer):
                    break

                if buffer[position] == ",":
                    position += 1
                    continue

                if buffer[position] == "]":
                    finished = True
                    position += 1
                    break

                try:
                    item, next_position = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break

                yield item
                position = next_position

            buffer = buffer[position:]

        trailing = buffer.strip()
        if trailing and trailing != "]":
            raise ValueError(f"Unexpected trailing content in {path}")
        if not finished and trailing != "]":
            raise ValueError(f"Incomplete JSON array in {path}")


class JsonArrayWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")
        self.handle.write("[\n")
        self.first = True

    def write(self, item):
        if not self.first:
            self.handle.write(",\n")
        json.dump(item, self.handle, indent=2, ensure_ascii=False)
        self.first = False

    def close(self):
        self.handle.write("\n]\n")
        self.handle.close()


def build_bucket_specs(boundaries):
    values = sorted(set(boundaries))
    bucket_specs = []
    lower = 0
    for upper in values:
        bucket_specs.append((f"{lower}-{upper}", lower, upper))
        lower = upper
    bucket_specs.append((f"{lower}-*", lower, None))
    return bucket_specs


def compute_length(tokenizer, text: str):
    return len(tokenizer.tokenize(text)) + tokenizer.num_special_tokens_to_add(pair=False)


def find_bucket(length: int, bucket_specs):
    for bucket_name, lower, upper in bucket_specs:
        if upper is None and length >= lower:
            return bucket_name
        if upper is not None and lower <= length < upper:
            return bucket_name
    raise ValueError(f"Unable to bucket sample with length {length}")


def prepare_directories(dataset_dir: Path, source_dirname: str, backup_dirname: str):
    source_dir = dataset_dir / source_dirname
    backup_dir = dataset_dir / backup_dirname

    if backup_dir.exists():
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Backup directory exists but output directory is missing: {source_dir}"
            )
        print(f"Resuming from existing backup directory: {backup_dir}")
        return source_dir, backup_dir

    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    source_dir.rename(backup_dir)
    source_dir.mkdir(parents=True, exist_ok=False)
    return source_dir, backup_dir


def cleanup_backup_dir(backup_dir: Path):
    if backup_dir.exists():
        shutil.rmtree(backup_dir)


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    dataset_dir = Path(args.dataset_dir).resolve()
    tokenizer_path = (
        Path(args.tokenizer_path).resolve()
        if args.tokenizer_path
        else repo_root / "model" / "Llama-3.2-1B"
    )

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.model_max_length = 10**12
    bucket_specs = build_bucket_specs(args.bucket_boundaries)
    split_names = tuple(args.splits)

    print(f"Using tokenizer: {tokenizer_path}")
    print(f"Dataset directory: {dataset_dir}")

    output_dir, input_dir = prepare_directories(
        dataset_dir, args.source_dirname, args.backup_dirname
    )

    print(f"Backup directory: {input_dir}")
    print(f"New output directory: {output_dir}")

    try:
        for split_name in split_names:
            input_path = input_dir / f"{args.input_prefix}_{split_name}.json"
            output_paths = {
                bucket_name: output_dir / f"{args.dataset_name}_{bucket_name}_{split_name}.json"
                for bucket_name, _, _ in bucket_specs
            }
            writers = {
                bucket_name: JsonArrayWriter(path)
                for bucket_name, path in output_paths.items()
            }
            bucket_counts = {bucket_name: 0 for bucket_name, _, _ in bucket_specs}
            total_count = 0
            max_item = None

            for row in iter_json_array(input_path):
                text = row.get("input", "")
                length = compute_length(tokenizer, text)
                enriched_row = dict(row)
                enriched_row["token_length"] = length
                bucket_name = find_bucket(length, bucket_specs)
                writers[bucket_name].write(enriched_row)
                bucket_counts[bucket_name] += 1
                total_count += 1

                if max_item is None or length > max_item[1]:
                    max_item = (row.get("index"), length)

            for writer in writers.values():
                writer.close()

            print(f"\nSplit: {split_name}")
            print(f"  total: {total_count}")
            print(f"  max: {max_item}")

            for bucket_name, _, _ in bucket_specs:
                print(f"  {bucket_name}: {bucket_counts[bucket_name]} -> {output_paths[bucket_name]}")

        cleanup_backup_dir(input_dir)
        print(f"\nDeleted backup directory: {input_dir}")
    except Exception:
        print(
            f"\nProcessing failed. The backup is still available at: {input_dir}"
        )
        raise


if __name__ == "__main__":
    main()
