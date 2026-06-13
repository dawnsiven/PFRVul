# OOF Reviewer Workflow

This document describes the full reviewer fine-tuning workflow when reviewer `train` is built from out-of-fold (OOF) small-model predictions, while reviewer `val` and `test` remain unchanged.

## Goal

- reviewer `train` uses OOF `reviewer_train.csv`
- reviewer `val` uses the original reviewer validation split
- reviewer `test` uses the original reviewer test split

## Overall Data Flow

1. Train the small model normally and keep its checkpoint.
2. Export the original reviewer CSV files from that checkpoint:
   `reviewer_train.csv`, `reviewer_val.csv`, and `reviewer_test.csv`.
3. Run OOF on the small-model training split and merge all held-out predictions into a new OOF `reviewer_train.csv`.
4. Build reviewer JSON files by replacing only reviewer `train` with the OOF version, while keeping the original reviewer `val` and `test`.
5. Rebucket reviewer `train/val/test` by token length.
6. Fine-tune the LoRA reviewer on OOF reviewer `train` and original reviewer `val`.
7. Run reviewer inference on the unchanged original reviewer `test`.

In short:

- original reviewer `val/test` come from the standard small-model reviewer export
- reviewer `train` comes from OOF merging
- LoRA training uses `OOF train + original val`
- LoRA inference uses `original test`

## Step 0. Prepare the Original Reviewer CSV Files

Before using OOF reviewer `train`, you still need the original reviewer `val` and `test` CSV files produced from the standard small-model checkpoint, because the OOF workflow replaces only reviewer `train`.

First, train the small model normally if you have not done so yet:

```shell
# Imbalance example:
./scripts/train_imbalance.sh bigvul_cwe20 CodeBERT 1 1 0

# Non-imbalance example:
./scripts/train.sh cvefixes_cwe352 CodeBERT 0-512 0
```

Then export the original reviewer CSV files from the trained checkpoint.

Use [`scripts/test_imbalance_test.sh`](scripts/test_imbalance_test.sh):

```shell
# Imbalance workflow:
./scripts/test_imbalance_test.sh bigvul_cwe20 CodeBERT 1 1 0

# Non-imbalance workflow:
./scripts/test_imbalance_test.sh cvefixes_cwe352 CodeBERT 0-512 0
```

Arguments:

- imbalance datasets:
  `<DATASET_NAME> <MODEL_NAME> <LENGTH> <POS_RATIO> [CUDA]`
- non-imbalance datasets:
  `<DATASET_NAME> <MODEL_NAME> <LENGTH> [CUDA]`

This script generates:

- `reviewer_train.csv`
- `reviewer_val.csv`
- `reviewer_test.csv`

under:

- imbalance:
  `outputs/<MODEL_NAME>_imbalance/<DATASET>_<LENGTH>_<POS_RATIO>/`
- non-imbalance:
  `outputs/<MODEL_NAME>/<DATASET>_<LENGTH>/`

These original reviewer CSV files are the source of reviewer `val/test` in the OOF workflow.

## Step 1. Generate OOF `reviewer_train.csv`

Use [`scripts/train_imbalance_oof.sh`](scripts/train_imbalance_oof.sh):

```shell
# Imbalance workflow:
OOF_RUN_TAG=myrun_eval_on_original_val \
./scripts/train_imbalance_oof.sh bigvul_cwe20 CodeBERT 1 1 0

# Non-imbalance workflow:
OOF_RUN_TAG=myrun_eval_on_original_val \
./scripts/train_imbalance_oof.sh cvefixes_cwe352 CodeBERT 0-512 0
```

Arguments:

- imbalance datasets:
  `<DATASET_NAME> <MODEL_NAME> <LENGTH> <POS_RATIO> [CUDA]`
- non-imbalance datasets:
  `<DATASET_NAME> <MODEL_NAME> <LENGTH> [CUDA]`

Environment variables:

- `OOF_FOLDS`: number of folds, default `5`
- `OOF_SEED`: fold split seed, default `42`
- `OOF_RUN_TAG`: output subdirectory name

This script will:

- split the original small-model training JSON into folds
- train one small model per fold on the other folds
- use the original `validate.json` as `eval_data_file` for checkpoint selection
- predict only on the held-out fold
- merge all held-out predictions into one OOF reviewer train CSV

This OOF file is used only to replace reviewer `train`. It does not replace reviewer `val` or reviewer `test`.

Important outputs:

- imbalance:
  `outputs/<MODEL_NAME>_imbalance_oof/<DATASET>_<LENGTH>_<POS_RATIO>/<OOF_RUN_TAG>/reviewer_train.csv`
- imbalance:
  `outputs/<MODEL_NAME>_imbalance_oof/<DATASET>_<LENGTH>_<POS_RATIO>/<OOF_RUN_TAG>/oof_summary.json`
- non-imbalance:
  `outputs/<MODEL_NAME>_oof/<DATASET>_<LENGTH>/<OOF_RUN_TAG>/reviewer_train.csv`
- non-imbalance:
  `outputs/<MODEL_NAME>_oof/<DATASET>_<LENGTH>/<OOF_RUN_TAG>/oof_summary.json`

Examples:

- imbalance:
  `outputs/CodeBERT_imbalance_oof/bigvul_cwe20_1_1/myrun_eval_on_original_val/reviewer_train.csv`
- non-imbalance:
  `outputs/CodeBERT_oof/cvefixes_cwe352_0-512/myrun_eval_on_original_val/reviewer_train.csv`

## Step 2. Build Reviewer JSON Files with OOF `train` and Original `val/test`

Use [`data_process/rebucket_reviewer_data_oof_train.sh`](data_process/rebucket_reviewer_data_oof_train.sh):

```shell
# Imbalance workflow:
./data_process/rebucket_reviewer_data_oof_train.sh bigvul_cwe20 CodeBERT 1 1

# Non-imbalance workflow:
./data_process/rebucket_reviewer_data_oof_train.sh cvefixes_cwe352 CodeBERT 0-512
```

Arguments:

- imbalance datasets:
  `<DATASET_NAME> <RESULT_MODEL_NAME> <LENGTH> <POS_RATIO> [OOF_RUN_TAG|OOF_TRAIN_CSV]`
- non-imbalance datasets:
  `<DATASET_NAME> <RESULT_MODEL_NAME> <LENGTH> [OOF_RUN_TAG|OOF_TRAIN_CSV]`

Default behavior:

- if the dataset uses imbalance mode with `POS_RATIO`, the script searches under:
  `outputs/<RESULT_MODEL_NAME>_imbalance_oof/<DATASET>_<LENGTH>_<POS_RATIO>/`
- if the dataset uses non-imbalance mode without `POS_RATIO`, the script searches under:
  `outputs/<RESULT_MODEL_NAME>_oof/<DATASET>_<LENGTH>/`
- if the optional OOF source argument is omitted, the script automatically finds the latest
  `reviewer_train.csv` under the corresponding OOF root directory
- if the optional OOF source argument is a run tag such as `myrun_eval_on_original_val`, the script resolves:
  `outputs/<..._oof>/<DATASET_TAG>/<OOF_RUN_TAG>/reviewer_train.csv`
- if the optional OOF source argument is a full CSV path, the script uses that file directly

Examples:

```shell
# Imbalance: auto-detect the latest OOF reviewer_train.csv.
./data_process/rebucket_reviewer_data_oof_train.sh bigvul_cwe20 CodeBERT 1 1

# Imbalance: use a specific OOF run tag.
./data_process/rebucket_reviewer_data_oof_train.sh bigvul_cwe20 CodeBERT 1 1 myrun_eval_on_original_val

# Non-imbalance: auto-detect the latest OOF reviewer_train.csv.
./data_process/rebucket_reviewer_data_oof_train.sh cvefixes_cwe352 CodeBERT 0-512

# Non-imbalance: use a specific OOF run tag.
./data_process/rebucket_reviewer_data_oof_train.sh cvefixes_cwe352 CodeBERT 0-512 myrun_eval_on_original_val
```

This script will:

- first call `scripts/generate_reviewer_finetune_json.py` to generate the original reviewer `train.json`, `val.json`, and `test.json`
- replace only the generated reviewer `train.json` with the converted OOF reviewer train data
- keep the original `val.json` and `test.json`
- call `scripts/rebucket_reviewer_json_by_length.py` to regenerate the bucketed JSON files

After this step:

- `train.json` comes from OOF `reviewer_train.csv`
- `val.json` comes from the original reviewer validation CSV
- `test.json` comes from the original reviewer test CSV

Important directories:

- imbalance source JSON files:
  `reviewer_finetune_data/<RESULT_MODEL_NAME>_imbalance/<DATASET>_<LENGTH>_<POS_RATIO>/`
- imbalance rebucketed JSON files:
  `reviewer_finetune_data/<RESULT_MODEL_NAME>_imbalance/<DATASET>_<LENGTH>_<POS_RATIO>_length_rebucketed/`
- non-imbalance source JSON files:
  `reviewer_finetune_data/<RESULT_MODEL_NAME>/<DATASET>_<LENGTH>/`
- non-imbalance rebucketed JSON files:
  `reviewer_finetune_data/<RESULT_MODEL_NAME>/<DATASET>_<LENGTH>_length_rebucketed/`

## Step 3. Run Reviewer LoRA Fine-Tuning

Use [`scripts/finetune_test.sh`](scripts/finetune_test.sh):

```shell
# Imbalance workflow:
./scripts/finetune_test.sh bigvul_cwe20 CodeBERT llama3.2 1 1 4 0-512 0

# Non-imbalance workflow:
./scripts/finetune_test.sh cvefixes_cwe352 CodeBERT llama3.2 0-512 4 0-512 0
```

Arguments:

- imbalance datasets:
  `<DATASET_NAME> <RESULT_MODEL_NAME> <LLM_MODEL_NAME> <LENGTH> <POS_RATIO> <BATCH_SIZE> [LENGTH_BUCKET] [CUDA]`
- non-imbalance datasets:
  `<DATASET_NAME> <RESULT_MODEL_NAME> <LLM_MODEL_NAME> <LENGTH> <BATCH_SIZE> [LENGTH_BUCKET] [CUDA]`

This script reads:

- imbalance:
  `reviewer_finetune_data/<RESULT_MODEL_NAME>_imbalance/<DATASET>_<LENGTH>_<POS_RATIO>_length_rebucketed/train_<LENGTH_BUCKET>.json`
- imbalance:
  `reviewer_finetune_data/<RESULT_MODEL_NAME>_imbalance/<DATASET>_<LENGTH>_<POS_RATIO>_length_rebucketed/val_<LENGTH_BUCKET>.json`
- non-imbalance:
  `reviewer_finetune_data/<RESULT_MODEL_NAME>/<DATASET>_<LENGTH>_length_rebucketed/train_<LENGTH_BUCKET>.json`
- non-imbalance:
  `reviewer_finetune_data/<RESULT_MODEL_NAME>/<DATASET>_<LENGTH>_length_rebucketed/val_<LENGTH_BUCKET>.json`

So after Step 2, LoRA fine-tuning automatically uses:

- OOF reviewer `train`
- original reviewer `val`

There is no additional data merging in this LoRA step. The OOF merging is already completed in Step 1.

## Step 4. Run Reviewer Inference on the Unchanged Reviewer Test Set

Use [`scripts/inference_finetune_test.sh`](scripts/inference_finetune_test.sh):

```shell
# Imbalance workflow:
./scripts/inference_finetune_test.sh bigvul_cwe20 CodeBERT llama3.2 1 1 0-512 0

# Non-imbalance workflow:
./scripts/inference_finetune_test.sh cvefixes_cwe352 CodeBERT llama3.2 0-512 0-512 0
```

Arguments:

- imbalance datasets:
  `<DATASET_NAME> <RESULT_MODEL_NAME> <LLM_MODEL_NAME> <LENGTH> <POS_RATIO> [LENGTH_BUCKET] [EPOCH|epoch-N] [CUDA]`
- non-imbalance datasets:
  `<DATASET_NAME> <RESULT_MODEL_NAME> <LLM_MODEL_NAME> <LENGTH> [LENGTH_BUCKET] [EPOCH|epoch-N] [CUDA]`

This script reads:

- imbalance:
  `reviewer_finetune_data/<RESULT_MODEL_NAME>_imbalance/<DATASET>_<LENGTH>_<POS_RATIO>_length_rebucketed/test_<LENGTH_BUCKET>.json`
- non-imbalance:
  `reviewer_finetune_data/<RESULT_MODEL_NAME>/<DATASET>_<LENGTH>_length_rebucketed/test_<LENGTH_BUCKET>.json`

So after Step 2, reviewer inference automatically uses:

- original reviewer `test`

After reviewer inference finishes, the workflow can also merge the reviewer result back into the original small-model `results.csv` and compute final metrics on the full test set.

This final merge is different from the OOF merge in Step 1:

- Step 1 OOF merge:
  merge the held-out fold predictions into one OOF `reviewer_train.csv`
- Final test-time merge:
  merge reviewer test predictions back into the original small-model `results.csv`

The final test-time merge is used for end-to-end evaluation of the reviewer pipeline. It answers questions such as:

- whether the reviewer improves final precision / recall / F1 on the full test set
- how many original small-model positive predictions were changed by the reviewer
- how the reviewed positive subset changes before and after reviewer correction

Current implementation:

- [`scripts/inference_finetune_test.sh`](scripts/inference_finetune_test.sh)
  runs reviewer inference and then automatically calls
  [`scripts/merge_reviewer_lora_results.py`](scripts/merge_reviewer_lora_results.py)
- [`scripts/merge_reviewer_lora_results.py`](scripts/merge_reviewer_lora_results.py)
  merges reviewer predictions back into the original small-model `results.csv`
  and writes:
  - `merged_results.csv`
  - `merged_metrics.json`

Important note:

- OOF is used only to build a cleaner reviewer `train`
- final result merging and metric analysis happen only after reviewer inference on the original reviewer `test`
- to merge correctly, reviewer inference results must preserve the original sample `Index`

Output files after Step 4:

- `results.csv`
  reviewer predictions on the reviewer test subset, כלומר the samples originally predicted as positive by the small model
- `merged_results.csv`
  final predictions after merging reviewer decisions back into the original small-model `results.csv`
- `merged_metrics.json`
  final merged metrics on the full test set, plus reviewed-subset metrics before and after reviewer correction

Compatibility note:

- old reviewer `results.csv` files generated before the `LLM/inference.py` index fix may use row order (`0, 1, 2, ...`) instead of the original sample `Index`
- those old files are not reliable for final merge evaluation
- for correct final merge results, rerun [`scripts/inference_finetune_test.sh`](scripts/inference_finetune_test.sh) after the index-fix update

## One-Command Pipeline Wrappers

If you do not want to run Step 0 through Step 4 manually, you can use the wrapper scripts below.

Single-dataset wrapper:

- [`scripts/run_reviewer_oof_pipeline.sh`](scripts/run_reviewer_oof_pipeline.sh)
  runs the full OOF workflow for one dataset by chaining:
  small-model train, reviewer CSV export, OOF generation, OOF reviewer JSON preparation,
  reviewer LoRA fine-tuning, reviewer inference, and final merge.

Examples:

```shell
# Imbalance workflow:
./scripts/run_reviewer_oof_pipeline.sh bigvul_cwe20 CodeBERT llama3.2 1 1 4 0-512 0 myrun

# Non-imbalance workflow:
./scripts/run_reviewer_oof_pipeline.sh cvefixes_cwe352 CodeBERT llama3.2 0-512 4 0-512 0 myrun
```

Multi-dataset wrapper:

- [`scripts/run_oof_full_pipeline.sh`](scripts/run_oof_full_pipeline.sh)
  reuses `scripts/run_reviewer_oof_pipeline.sh` and executes the same full workflow
  sequentially for multiple datasets.

Usage rule:

- pass one dataset name, or multiple dataset names separated by commas, as the first argument
- all remaining arguments are forwarded unchanged to
  `scripts/run_reviewer_oof_pipeline.sh`
- this is useful when several datasets share the same
  `RESULT_MODEL_NAME`, `LLM_MODEL_NAME`, `LENGTH`, `POS_RATIO`, `BATCH_SIZE`,
  `LENGTH_BUCKET`, `CUDA`, and `OOF_RUN_TAG`

Examples:

```shell
# One dataset:
./scripts/run_oof_full_pipeline.sh cvefixes_cwe352 CodeBERT llama3.2 0-512 4

# Multiple non-imbalance datasets:
./scripts/run_oof_full_pipeline.sh cvefixes_cwe352,cvefixes_cwe79 CodeBERT llama3.2 0-512 4

# Multiple imbalance datasets:
./scripts/run_oof_full_pipeline.sh bigvul_cwe20,bigvul_cwe79 CodeBERT llama3.2 1 1 4 0-512 0 batch_run
```

## Related Scripts

- [`scripts/train_imbalance_test.sh`](scripts/train_imbalance_test.sh)
  Export original reviewer `train/val/test` CSV files from an existing small-model checkpoint on imbalance data.

- [`scripts/train_imbalance_oof.sh`](scripts/train_imbalance_oof.sh)
  Generate OOF `reviewer_train.csv` by fold-based training on the small model.

- [`scripts/generate_reviewer_finetune_json.py`](scripts/generate_reviewer_finetune_json.py)
  Convert reviewer CSV files into reviewer `train.json`, `val.json`, and `test.json`.

- [`data_process/rebucket_reviewer_data_oof_train.sh`](data_process/rebucket_reviewer_data_oof_train.sh)
  Replace reviewer `train` with OOF train and keep original reviewer `val/test`.

- [`scripts/rebucket_reviewer_json_by_length.py`](scripts/rebucket_reviewer_json_by_length.py)
  Rebucket reviewer JSON files into per-length-bucket train/val/test JSON files.

- [`scripts/finetune_test.sh`](scripts/finetune_test.sh)
  Run reviewer LoRA fine-tuning.

- [`scripts/inference_finetune_test.sh`](scripts/inference_finetune_test.sh)
  Run reviewer inference on the reviewer test split.

- [`scripts/merge_reviewer_lora_results.py`](scripts/merge_reviewer_lora_results.py)
  Merge reviewer LoRA test results back into the original small-model `results.csv`
  and compute final merged metrics.

- [`scripts/run_reviewer_oof_pipeline.sh`](scripts/run_reviewer_oof_pipeline.sh)
  Run the full OOF reviewer pipeline for a single dataset.

- [`scripts/run_oof_full_pipeline.sh`](scripts/run_oof_full_pipeline.sh)
  Run the same full OOF reviewer pipeline sequentially for one or more datasets.
