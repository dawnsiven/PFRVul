from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

INPUT_GROUPS = REPO_ROOT / "LLM_TEST" / "intermediate" / "chunk_lable_0-512_chunk_review_per_chunk" / "chunk_review_groups.json"
PAIRS_JSONL = REPO_ROOT / "data" / "chunk_lable" / "pairs" / "cvefixes_local_fix_pairs_test.jsonl"
OUTPUT_JSON = REPO_ROOT / "LLM_TEST" / "intermediate" / "chunk_lable_selected_cases.json"

MODEL_RESULTS = {
    "deepseek": REPO_ROOT / "LLM_TEST" / "output" / "chunk_lable_0-512_chunk_review_per_chunk_full_deepseek" / "chunk_review_results.json",
    "mimo": REPO_ROOT / "LLM_TEST" / "output" / "chunk_lable_0-512_chunk_review_per_chunk_full_mimo" / "chunk_review_results.json",
    "moonshot": REPO_ROOT / "LLM_TEST" / "output" / "chunk_lable_0-512_chunk_review_per_chunk_full_moonshot" / "chunk_review_results.json",
}

SELECTED_INDEXES = [1278, 1101, 888, 784, 189]


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pairs(path: Path) -> dict[str, dict]:
    pairs: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            pair_id = str(record.get("pair_id", "")).strip()
            if pair_id:
                pairs[pair_id] = record
    return pairs


def build_case_bundle() -> dict:
    groups_payload = load_json(INPUT_GROUPS)
    assert isinstance(groups_payload, dict) and isinstance(groups_payload.get("groups"), list)
    groups_by_index = {int(group["index"]): group for group in groups_payload["groups"]}

    pairs_by_id = load_pairs(PAIRS_JSONL)

    model_rows: dict[str, dict[int, dict]] = {}
    for model_name, path in MODEL_RESULTS.items():
        payload = load_json(path)
        assert isinstance(payload, dict) and isinstance(payload.get("results"), list)
        model_rows[model_name] = {int(row["index"]): row for row in payload["results"]}

    cases: list[dict] = []
    for index in SELECTED_INDEXES:
        group = groups_by_index[index]
        pair_id = str(group["metadata"]["pair_id"])
        pair_record = pairs_by_id[pair_id]

        model_results: dict[str, dict] = {}
        votes: dict[str, int | None] = {}
        for model_name, rows in model_rows.items():
            row = rows[index]
            result = row.get("result", {})
            parsed = result.get("parsed_response", {})
            prediction = result.get("prediction")
            votes[model_name] = prediction
            model_results[model_name] = {
                "prediction": prediction,
                "is_vulnerable": result.get("is_vulnerable"),
                "parse_status": result.get("parse_status"),
                "cwe": parsed.get("cwe", ""),
                "reason": parsed.get("reason", ""),
                "raw_response": result.get("raw_response", ""),
            }

        positive_votes = sum(1 for vote in votes.values() if vote == 1)
        negative_votes = sum(1 for vote in votes.values() if vote == 0)
        majority_prediction = None
        if positive_votes > negative_votes:
            majority_prediction = 1
        elif negative_votes > positive_votes:
            majority_prediction = 0

        cases.append(
            {
                "sample": {
                    "index": group["index"],
                    "pair_id": pair_id,
                    "group_id": group["group_id"],
                    "filename": group["filename"],
                    "language": group["language"],
                    "cve_id": group["metadata"]["cve_id"],
                    "cwe_id": group["metadata"]["cwe_id"],
                    "method_name": group["metadata"]["method_name"],
                    "method_signature": group["metadata"]["method_signature"],
                    "side": group["metadata"]["side"],
                    "ground_truth": group["ground_truth"],
                    "original_small_model_prediction": group["original_prediction"],
                    "code_length": len(group["code"]),
                    "chunk_length": len(group["chunks"][0]["text"]),
                    "extra_chars": len(group["code"]) - len(group["chunks"][0]["text"]),
                    "selection_tag": (
                        "negative_majority_correct"
                        if int(group["ground_truth"]) == 0
                        else "positive_consensus_or_majority_correct"
                    ),
                },
                "chunk": group["chunks"][0],
                "code_before": (pair_record.get("before") or {}).get("code", ""),
                "code_after": (pair_record.get("after") or {}).get("code", ""),
                "model_results": model_results,
                "majority_vote": {
                    "votes": votes,
                    "positive_votes": positive_votes,
                    "negative_votes": negative_votes,
                    "majority_prediction": majority_prediction,
                    "majority_correct": majority_prediction == group["ground_truth"],
                },
            }
        )

    return {
        "description": (
            "Selected case samples for paper writing. Includes four positive samples "
            "and one previously selected negative sample."
        ),
        "models": list(MODEL_RESULTS.keys()),
        "cases": cases,
    }


def main() -> None:
    payload = build_case_bundle()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()
