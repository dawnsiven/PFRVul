import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract hidden-state representations for code samples without modifying existing model code."
    )
    parser.add_argument("--model-type", choices=["codebert", "unixcoder"], required=True)
    parser.add_argument("--model-name-or-path", required=True, help="Tokenizer/base model name.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint-best-f1/model.bin.")
    parser.add_argument("--data-json", required=True, help="Path to source JSON dataset.")
    parser.add_argument("--metadata-csv", required=True, help="Path to save metadata CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory to save .npy embeddings.")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument(
        "--pooling",
        nargs="+",
        default=["cls", "mean"],
        choices=["cls", "mean", "mean_no_special"],
        help="Pooling methods to export.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_model(model_name_or_path: str, model_type: str):
    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    config = RobertaConfig.from_pretrained(model_name_or_path, local_files_only=True)
    config.num_labels = 1 if model_type == "codebert" else 2
    model = RobertaForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def load_checkpoint(model, checkpoint_path: Path):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            stripped[key[len("encoder."):]] = value
    if stripped:
        missing, unexpected = model.load_state_dict(stripped, strict=False)
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected


def normalize_code(text: str) -> str:
    return " ".join(text.split())


def encode_sample(tokenizer, model_type: str, code: str, block_size: int):
    code_tokens = tokenizer.tokenize(normalize_code(code))
    if model_type == "codebert":
        code_tokens = code_tokens[: block_size - 2]
        tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    else:
        code_tokens = code_tokens[: block_size - 4]
        tokens = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = input_ids + [tokenizer.pad_token_id] * (block_size - len(input_ids))
    attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in input_ids]
    special_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    return input_ids, attention_mask, special_mask


class JsonCodeDataset(Dataset):
    def __init__(self, tokenizer, model_type: str, data_json: Path, block_size: int):
        with data_json.open("r", encoding="utf-8") as handle:
            self.rows = json.load(handle)
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.block_size = block_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        sample_index = row.get("index", row.get("Index", index))
        label = row.get("output", row.get("Label", 0))
        input_ids, attention_mask, special_mask = encode_sample(
            self.tokenizer,
            self.model_type,
            row.get("input", ""),
            self.block_size,
        )
        return {
            "index": int(sample_index),
            "label": int(label),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "special_mask": torch.tensor(special_mask, dtype=torch.long),
            "input": row.get("input", ""),
        }


def collate_fn(batch):
    return {
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "special_mask": torch.stack([item["special_mask"] for item in batch]),
        "input": [item["input"] for item in batch],
    }


def masked_mean(hidden_states, mask):
    mask = mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-12)
    return summed / denom


def extract_embeddings(model, dataloader, device, pooling_methods):
    outputs = {name: [] for name in pooling_methods}
    metadata_rows = []

    model.to(device)
    model.eval()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        special_mask = batch["special_mask"].to(device)
        with torch.no_grad():
            encoder_outputs = model.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            last_hidden = encoder_outputs.last_hidden_state

        if "cls" in pooling_methods:
            outputs["cls"].append(last_hidden[:, 0, :].cpu().numpy())
        if "mean" in pooling_methods:
            outputs["mean"].append(masked_mean(last_hidden, attention_mask).cpu().numpy())
        if "mean_no_special" in pooling_methods:
            valid_mask = attention_mask * (1 - special_mask)
            outputs["mean_no_special"].append(masked_mean(last_hidden, valid_mask).cpu().numpy())

        for sample_index, label, code in zip(
            batch["index"].tolist(),
            batch["label"].tolist(),
            batch["input"],
        ):
            metadata_rows.append(
                {
                    "Index": sample_index,
                    "Label": label,
                    "input": code,
                }
            )

    outputs = {name: np.concatenate(chunks, axis=0) for name, chunks in outputs.items()}
    return outputs, metadata_rows


def write_metadata(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Index", "Label", "input"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    data_json = Path(args.data_json)
    metadata_csv = Path(args.metadata_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    tokenizer, model = build_model(args.model_name_or_path, args.model_type)
    missing, unexpected = load_checkpoint(model, checkpoint)

    dataset = JsonCodeDataset(tokenizer, args.model_type, data_json, args.block_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    embeddings, metadata_rows = extract_embeddings(model, dataloader, device, args.pooling)

    write_metadata(metadata_csv, metadata_rows)
    for pooling_name, array in embeddings.items():
        np.save(output_dir / f"{pooling_name}.npy", array)

    print(f"model_type={args.model_type}")
    print(f"checkpoint={checkpoint}")
    print(f"data_json={data_json}")
    print(f"device={device}")
    print(f"samples={len(dataset)}")
    print(f"pooling={','.join(args.pooling)}")
    print(f"metadata_csv={metadata_csv}")
    print(f"output_dir={output_dir}")
    if missing:
        print(f"missing_keys={len(missing)}")
    if unexpected:
        print(f"unexpected_keys={len(unexpected)}")


if __name__ == "__main__":
    main()
