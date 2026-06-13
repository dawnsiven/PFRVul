# `build_cvefixes_astchunk.py`

This document describes the purpose and usage of
[`build_cvefixes_astchunk.py`](/home/zjr123/CVEfixes_v1.0.8/build_cvefixes_astchunk.py).

## What It Does

The script builds Alpaca-style JSONL datasets directly from `Data/CVEfixes.db`.

It works at the `method_change` level:

- reads method code from `method_change.code`
- reads language and file metadata from `file_change`
- pairs vulnerable and fixed methods using `before_change`
- computes line-level diffs between the paired methods
- chunks code with ASTChunk when the language is supported
- falls back to fixed-line chunking when AST chunking is unavailable
- keeps only chunks that overlap changed lines
- writes Alpaca-style `train/validate/test` JSONL files

## Pairing Rules

Methods are paired inside the same `file_change_id`.

Pair key:

- prefer `file_change_id + signature`
- if `signature` is empty, fall back to `file_change_id + name`

Direction:

- `before_change=True` or `1` is treated as vulnerable (`output = "1"`)
- `before_change=False` or `0` is treated as fixed (`output = "0"`)

`unpaired_methods` in the log means a group did not contain a complete before/after pair.

Common reasons:

- a method exists only on one side of the patch
- method name or signature changed
- `--limit` cut the SQL result before the matching side was reached

## Chunking Behavior

The script tries AST-based chunking first.

Currently allowed AST languages:

- `python`
- `java`
- `csharp`
- `typescript`
- `cpp`
- `javascript`
- `php`

If ASTChunk or its parser dependencies are unavailable, the script falls back to:

1. fixed-line chunking using `--fallback_lines`
2. further splitting oversized chunks using tokenizer-based token count when available

The output file name includes the configured max token budget, for example:

- `cvefixes_0-512_train.jsonl`

This means the script used `--max_tokens 512`.
If `--tokenizer_model_path` can be loaded, the limit is measured with that tokenizer.
Otherwise the script falls back to approximate token counting.

## Output Layout

The script writes two kinds of outputs:

- all selected data:
  - `data/cvefixes/alpaca/...jsonl`
- per-CWE subsets:
  - `data/by_cwe/cvefixes_cwe77/alpaca/...jsonl`

If you run with `--cwes cwe77`, the full dataset output and the CWE-77 subset will be very similar,
because the selected input set only contains CWE-77 rows.

## Common Commands

Run a small smoke test:

```bash
python3 -u build_cvefixes_astchunk.py \
  --db Data/CVEfixes.db \
  --output_root data \
  --cwes cwe77 \
  --limit 200 \
  --progress_every 20 \
  --log_file data/cwe77_run.log
```

Run a full export for one CWE:

```bash
python3 -u build_cvefixes_astchunk.py \
  --db Data/CVEfixes.db \
  --output_root data \
  --cwes cwe77 \
  --progress_every 100 \
  --log_file data/cwe77_run.log
```

Run several CWEs together:

```bash
python3 -u build_cvefixes_astchunk.py \
  --db Data/CVEfixes.db \
  --output_root data \
  --cwes cwe77 cwe78 cwe79 \
  --progress_every 100 \
  --log_file data/cwe77_78_79_run.log
```

Keep the old flat CWE output layout:

```bash
python3 -u build_cvefixes_astchunk.py \
  --db Data/CVEfixes.db \
  --output_root data \
  --cwes cwe77 \
  --cwe_subdir "" \
  --log_file data/cwe77_run.log
```

## Important Arguments

- `--db`: SQLite database path, usually `Data/CVEfixes.db`
- `--output_root`: root output directory
- `--cwes`: one or more CWE filters, such as `cwe77`, `77`, `CWE-78`
- `--limit`: limit the SQL result rows for debugging
- `--max_tokens`: per-chunk token cap
- `--tokenizer_model_path`: tokenizer/model path used for chunk length counting
- `--max_chars`: ASTChunk max non-whitespace size
- `--fallback_lines`: line window size when fallback chunking is used
- `--progress_every`: log progress every N processed method rows
- `--log_file`: write progress and status logs to a file
- `--count_total`: run an extra `COUNT(*)` query to show percentage progress

## Notes

- `--limit` is applied before pairing is finished, so it can increase `unpaired_methods`.
- The script currently filters by selected CWE in SQL before joining into the larger tables.
- Output is written directly as JSONL.
  This avoids holding all records in RAM and fits the streaming batch design better.

## Related Files

- script:
  - [`build_cvefixes_astchunk.py`](/home/zjr123/CVEfixes_v1.0.8/build_cvefixes_astchunk.py)
- AST chunk builder:
  - [`astchunk-main/src/astchunk/astchunk_builder.py`](/home/zjr123/CVEfixes_v1.0.8/astchunk-main/src/astchunk/astchunk_builder.py)
- ASTChunk project README:
  - [`astchunk-main/README.md`](/home/zjr123/CVEfixes_v1.0.8/astchunk-main/README.md)
