import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity between code representations and export top-k neighbors."
    )
    parser.add_argument("--embeddings", required=True, help="Path to .npy embeddings file.")
    parser.add_argument("--metadata-csv", required=True, help="Path to metadata CSV aligned with embeddings.")
    parser.add_argument("--errors-csv", required=True, help="Path to collected error CSV.")
    parser.add_argument("--output-csv", required=True, help="Path to save top-k similarity pairs.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--compare-mode",
        choices=["error_vs_all", "error_vs_error"],
        default="error_vs_all",
    )
    return parser.parse_args()


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def build_error_map(error_rows):
    error_map = {}
    for row in error_rows:
        error_map[int(row["Index"])] = row
    return error_map


def build_metadata_map(metadata_rows):
    metadata_map = {}
    for idx, row in enumerate(metadata_rows):
        metadata_map[int(row["Index"])] = {
            "row_id": idx,
            "Label": row.get("Label", ""),
            "input": row.get("input", ""),
        }
    return metadata_map


def topk_neighbors(similarities, k, forbidden_index):
    order = np.argsort(-similarities)
    result = []
    for idx in order:
        if int(idx) == forbidden_index:
            continue
        result.append((int(idx), float(similarities[idx])))
        if len(result) >= k:
            break
    return result


def main():
    args = parse_args()
    embeddings = np.load(args.embeddings)
    metadata_rows = load_csv(Path(args.metadata_csv))
    error_rows = load_csv(Path(args.errors_csv))

    if len(embeddings) != len(metadata_rows):
        raise ValueError("Embeddings row count must match metadata CSV row count.")

    normalized = normalize_rows(embeddings)
    metadata_map = build_metadata_map(metadata_rows)
    error_map = build_error_map(error_rows)

    all_indices = [int(row["Index"]) for row in metadata_rows]
    error_indices = [int(row["Index"]) for row in error_rows if int(row["Index"]) in metadata_map]
    if args.compare_mode == "error_vs_error":
        candidate_indices = error_indices
    else:
        candidate_indices = all_indices

    output_rows = []
    for query_index in error_indices:
        query_row_id = metadata_map[query_index]["row_id"]
        candidate_row_ids = [metadata_map[idx]["row_id"] for idx in candidate_indices]
        similarity_scores = normalized[query_row_id] @ normalized[candidate_row_ids].T
        neighbors = topk_neighbors(similarity_scores, args.top_k, candidate_row_ids.index(query_row_id) if query_index in candidate_indices else -1)

        for local_neighbor_pos, similarity in neighbors:
            neighbor_row_id = candidate_row_ids[local_neighbor_pos]
            neighbor_index = all_indices[neighbor_row_id]
            neighbor_error = error_map.get(neighbor_index, {})
            query_error = error_map[query_index]
            output_rows.append(
                {
                    "query_index": query_index,
                    "query_label": query_error.get("Label", ""),
                    "query_prediction": query_error.get("Prediction", ""),
                    "query_error_type": query_error.get("error_type", ""),
                    "neighbor_index": neighbor_index,
                    "neighbor_label": metadata_rows[neighbor_row_id].get("Label", ""),
                    "neighbor_prediction": neighbor_error.get("Prediction", ""),
                    "neighbor_error_type": neighbor_error.get("error_type", "correct"),
                    "similarity": similarity,
                }
            )

    write_csv(
        Path(args.output_csv),
        output_rows,
        fieldnames=[
            "query_index",
            "query_label",
            "query_prediction",
            "query_error_type",
            "neighbor_index",
            "neighbor_label",
            "neighbor_prediction",
            "neighbor_error_type",
            "similarity",
        ],
    )

    print(f"embeddings={args.embeddings}")
    print(f"metadata_csv={args.metadata_csv}")
    print(f"errors_csv={args.errors_csv}")
    print(f"compare_mode={args.compare_mode}")
    print(f"errors={len(error_indices)}")
    print(f"top_k={args.top_k}")
    print(f"output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
