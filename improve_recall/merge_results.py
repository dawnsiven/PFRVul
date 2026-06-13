import argparse
import csv
from pathlib import Path


DEFAULT_CODEBERT_ROOT = Path("outputs/CodeBERT_imbalance")
DEFAULT_UNIXCODER_ROOT = Path("outputs/UniXcoder_imbalance")
DEFAULT_OUTPUT_ROOT = Path("improve_recall/output")


def normalize_row(row, model_name):
    return {
        "Index": str(row["Index"]).strip(),
        "Label": int(row["Label"]),
        "Prediction": int(row["Prediction"]),
        "Prob": float(row.get("Prob", row.get("prob", 0.0))),
        "CWE": row.get("CWE", "null"),
        "Model": model_name,
    }


def load_results(csv_path, model_name):
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [normalize_row(row, model_name) for row in reader]
    return {row["Index"]: row for row in rows}


def combine_prediction(codebert_pred, unixcoder_pred, strategy):
    if strategy == "union":
        return int(codebert_pred == 1 or unixcoder_pred == 1)
    if strategy == "intersection":
        return int(codebert_pred == 1 and unixcoder_pred == 1)
    if strategy == "codebert":
        return int(codebert_pred)
    if strategy == "unixcoder":
        return int(unixcoder_pred)
    raise ValueError(f"Unsupported strategy: {strategy}")


def combine_prob(codebert_prob, unixcoder_prob, strategy):
    if strategy in {"union", "intersection"}:
        return max(codebert_prob, unixcoder_prob)
    if strategy == "codebert":
        return codebert_prob
    if strategy == "unixcoder":
        return unixcoder_prob
    raise ValueError(f"Unsupported strategy: {strategy}")


def merge_results(dataset_name, codebert_root, unixcoder_root, output_root, strategy):
    codebert_csv = codebert_root / dataset_name / "results.csv"
    unixcoder_csv = unixcoder_root / dataset_name / "results.csv"

    if not codebert_csv.exists():
        raise FileNotFoundError(f"CodeBERT results not found: {codebert_csv}")
    if not unixcoder_csv.exists():
        raise FileNotFoundError(f"UniXcoder results not found: {unixcoder_csv}")

    codebert_rows = load_results(codebert_csv, "CodeBERT")
    unixcoder_rows = load_results(unixcoder_csv, "UniXcoder")

    if set(codebert_rows) != set(unixcoder_rows):
        missing_in_unixcoder = sorted(set(codebert_rows) - set(unixcoder_rows))
        missing_in_codebert = sorted(set(unixcoder_rows) - set(codebert_rows))
        raise ValueError(
            "Index mismatch between model outputs. "
            f"Missing in UniXcoder: {missing_in_unixcoder[:10]}; "
            f"Missing in CodeBERT: {missing_in_codebert[:10]}"
        )

    output_dir = output_root / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "result.csv"

    fieldnames = [
        "Index",
        "CWE",
        "Label",
        "CodeBERT_Prediction",
        "CodeBERT_Prob",
        "UniXcoder_Prediction",
        "UniXcoder_Prob",
        "Final_Prediction",
        "Final_Prob",
        "Strategy",
    ]

    ordered_indices = sorted(codebert_rows, key=lambda item: int(item) if item.isdigit() else item)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index in ordered_indices:
            codebert_row = codebert_rows[index]
            unixcoder_row = unixcoder_rows[index]

            if codebert_row["Label"] != unixcoder_row["Label"]:
                raise ValueError(f"Label mismatch for Index={index}")

            cwe = unixcoder_row["CWE"]
            if cwe in {"", None}:
                cwe = "null"

            final_prediction = combine_prediction(
                codebert_row["Prediction"],
                unixcoder_row["Prediction"],
                strategy,
            )
            final_prob = combine_prob(
                codebert_row["Prob"],
                unixcoder_row["Prob"],
                strategy,
            )

            writer.writerow(
                {
                    "Index": index,
                    "CWE": cwe,
                    "Label": codebert_row["Label"],
                    "CodeBERT_Prediction": codebert_row["Prediction"],
                    "CodeBERT_Prob": codebert_row["Prob"],
                    "UniXcoder_Prediction": unixcoder_row["Prediction"],
                    "UniXcoder_Prob": unixcoder_row["Prob"],
                    "Final_Prediction": final_prediction,
                    "Final_Prob": final_prob,
                    "Strategy": strategy,
                }
            )

    return output_csv


def build_parser():
    parser = argparse.ArgumentParser(
        description="Merge CodeBERT and UniXcoder results for the same dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset folder name, for example: bigvul_cwe20_1_1",
    )
    parser.add_argument(
        "--codebert-root",
        type=Path,
        default=DEFAULT_CODEBERT_ROOT,
        help=f"Root directory of CodeBERT outputs. Default: {DEFAULT_CODEBERT_ROOT}",
    )
    parser.add_argument(
        "--unixcoder-root",
        type=Path,
        default=DEFAULT_UNIXCODER_ROOT,
        help=f"Root directory of UniXcoder outputs. Default: {DEFAULT_UNIXCODER_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Root directory for merged outputs. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--strategy",
        choices=["union", "intersection", "codebert", "unixcoder"],
        default="union",
        help="How to combine predictions. Default: union",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    output_csv = merge_results(
        dataset_name=args.dataset,
        codebert_root=args.codebert_root,
        unixcoder_root=args.unixcoder_root,
        output_root=args.output_root,
        strategy=args.strategy,
    )
    print(f"Merged results saved to: {output_csv}")


if __name__ == "__main__":
    main()
