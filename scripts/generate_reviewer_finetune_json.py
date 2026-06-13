#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


INSTRUCTION = (
    "The small model predicts that the following code contains a vulnerability. "
    "Determine whether this prediction should be kept or rejected."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reviewer finetuning JSON files from reviewer CSV outputs."
    )
    parser.add_argument("--dataset-name", required=True, help="Dataset name, e.g. bigvul_cwe20")
    parser.add_argument("--result-model", required=True, help="Small model name, e.g. CodeBERT")
    parser.add_argument("--length", required=True, help="Length tag, e.g. 1")
    parser.add_argument("--pos-ratio", default="", help="Positive ratio tag, e.g. 1")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root. Defaults to the parent of this script's directory.",
    )
    parser.add_argument(
        "--output-root",
        default="reviewer_finetune_data",
        help="Directory for generated reviewer JSON files.",
    )
    return parser.parse_args()


def load_source_samples(path: Path):
    records = json.loads(path.read_text(encoding="utf-8"))
    by_index = {}
    for record in records:
        sample_index = record.get("index")
        if sample_index is None:
            raise ValueError(f"Missing 'index' field in source dataset: {path}")
        by_index[int(sample_index)] = record
    return by_index


def resolve_results_dir(repo_root: Path, result_model: str, dataset_tag: str, imbalance: bool):
    if not imbalance:
        primary = repo_root / "outputs" / result_model / dataset_tag
        if primary.is_dir():
            return primary
        raise FileNotFoundError(
            "Could not find reviewer CSV directory. Expected: "
            f"{primary}"
        )

    primary = repo_root / "outputs" / f"{result_model}_imbalance" / dataset_tag
    fallback = repo_root / "outputs" / f"{result_model}_imbalance_test" / dataset_tag
    if primary.is_dir():
        return primary
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(
        "Could not find reviewer CSV directory. Expected one of: "
        f"{primary} or {fallback}"
    )


def build_prompt_input(prob_text: str, code: str):
    return f"Small model confidence: {prob_text}\n\nCode:\n{code}"


def convert_split(csv_path: Path, source_by_index, output_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Index", "Label", "Prediction", "prob"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

        for row in reader:
            prediction = int(row["Prediction"])
            if prediction != 1:
                continue

            sample_index = int(row["Index"])
            source_record = source_by_index.get(sample_index)
            if source_record is None:
                raise KeyError(f"Index {sample_index} from {csv_path} not found in source dataset")

            prob_text = row["prob"].strip()
            label = int(row["Label"])
            rows.append(
                {
                    "instruction": INSTRUCTION,
                    "input": build_prompt_input(prob_text, source_record["input"]),
                    "prob": float(prob_text),
                    "output": "1" if label == 1 else "0",
                    "index": sample_index,
                }
            )

    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(rows)


def main():
    args = parse_args()
    repo_root = (
        Path(args.repo_root).resolve()
        if args.repo_root
        else Path(__file__).resolve().parents[1]
    )
    imbalance = bool(args.pos_ratio)
    dataset_tag = (
        f"{args.dataset_name}_{args.length}_{args.pos_ratio}"
        if imbalance
        else f"{args.dataset_name}_{args.length}"
    )
    results_dir = resolve_results_dir(repo_root, args.result_model, dataset_tag, imbalance)

    dataset_dir = (
        repo_root / "data" / f"{args.dataset_name}_subsampled" / "alpaca"
        if imbalance
        else repo_root / "data" / args.dataset_name / "alpaca"
    )
    source_paths = {
        "train": dataset_dir / f"{dataset_tag}_train.json",
        "val": dataset_dir / f"{dataset_tag}_validate.json",
        "test": dataset_dir / f"{dataset_tag}_test.json",
    }
    csv_paths = {
        "train": results_dir / "reviewer_train.csv",
        "val": results_dir / "reviewer_val.csv",
        "test": results_dir / "reviewer_test.csv",
    }

    output_root = (repo_root / args.output_root).resolve()
    output_dir = (
        output_root / f"{args.result_model}_imbalance" / dataset_tag
        if imbalance
        else output_root / args.result_model / dataset_tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        split: output_dir / f"{split}.json"
        for split in ("train", "val", "test")
    }

    counts = {}
    for split in ("train", "val", "test"):
        if not source_paths[split].is_file():
            raise FileNotFoundError(f"Missing source dataset JSON: {source_paths[split]}")
        if not csv_paths[split].is_file():
            raise FileNotFoundError(f"Missing reviewer CSV: {csv_paths[split]}")

        source_by_index = load_source_samples(source_paths[split])
        counts[split] = convert_split(csv_paths[split], source_by_index, output_paths[split])

    print(f"Results directory: {results_dir}")
    for split in ("train", "val", "test"):
        print(f"{split}: {counts[split]} samples -> {output_paths[split]}")


if __name__ == "__main__":
    main()
