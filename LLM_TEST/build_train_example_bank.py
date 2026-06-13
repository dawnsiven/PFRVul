import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a train-split example bank for LLM retrieval from weighted or filtered train data."
    )
    parser.add_argument(
        "--train-json",
        default="data/bigvul_cwe20_subsampled/alpaca/bigvul_cwe20_1_1_train_weighted.json",
        help="Path to train json file.",
    )
    parser.add_argument(
        "--output-json",
        default="LLM_TEST/intermediate/bigvul_cwe20_1_1_train_example_bank.json",
        help="Path to save the filtered train example bank.",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=2.0,
        help="Only keep train samples whose weight is >= this threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_json = Path(args.train_json)
    output_json = Path(args.output_json)

    with train_json.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    kept = []
    for row in rows:
        weight = float(row.get("weight", 1.0))
        if weight < args.min_weight:
            continue
        kept.append(
            {
                "Index": int(row.get("index", 0)),
                "Label": int(row.get("output", 0)),
                "Prediction": int(row.get("output", 0)),
                "error_type": "train_weighted",
                "weight": weight,
                "input": row.get("input", ""),
            }
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(kept, handle, ensure_ascii=False, indent=2)

    print(f"train_json={train_json}")
    print(f"min_weight={args.min_weight}")
    print(f"kept={len(kept)}")
    print(f"output_json={output_json}")


if __name__ == "__main__":
    main()
