# LLM4CVD: An Experimental Study

This is the codebase for the paper "Investigating Large Language Models for Code Vulnerability Detection: An Experimental Study", which is submitted to IEEE TIFS. The arxiv version of this work is publicly available via [this link](https://arxiv.org/abs/2412.18260). We also provide the 🤗Huggingface repository for all evaluated datasets via [dataset link](https://huggingface.co/datasets/xuefen/VulResource) and all trained models via [model link](https://huggingface.co/Sak1Rinn/LLM4CVD-models).

To facilitate related communities and encourage future studies, we provide an easy-to-use and unified codebase to implement three graph-based models, two medium-size BERT-based sequence models, and four LLMs to study their performance for the code vulnerability detection task. Our codebase is built on the top of some related codebases provided below (Awesome Helpful Resources).


## Investigated Model List

| Dataset | Venue | Type | Paper Link |
| --- | --- | --- | --- |
| Devign | NeurIPS  | Graph | [Link](https://proceedings.neurips.cc/paper_files/paper/2019/hash/49265d2447bc3bbfe9e76306ce40a31f-Abstract.html) |
| ReGVD | IEEE ICSE  | Graph | [Link](https://dl.acm.org/doi/abs/10.1145/3510454.3516865) |
| GraphCodeBERT | ICLR  | Graph | [Link](https://arxiv.org/abs/2009.08366) |
| CodeBERT | EMNLP  | Sequence | [Link](https://arxiv.org/abs/2002.08155) |
| UniXcoder | ACL  | Sequence | [Link](https://arxiv.org/abs/2203.03850) |
| Llama-2-7B | Arxiv  | Sequence | [Link](https://arxiv.org/abs/2307.09288) |
| CodeLlama-7B | Arxiv  | Sequence | [Link](https://arxiv.org/abs/2308.12950) |
| Llama-3-8B | Arxiv  | Sequence | [Link](https://arxiv.org/abs/2407.21783) |
| Llama-3.1-8B | Arxiv  | Sequence | [Link](https://arxiv.org/abs/2407.21783) |


## Dataset

We provide our converted datasets in our HuggingFace dataset repository.
At the `linux` terminal of this current path, you can download the datasets by the following command, and remember to rename `VulResource` to `data`:

```shell
git clone https://huggingface.co/datasets/xuefen/VulResource
```

**Original paper and resources are listed below.**

| Dataset | Venue |  Paper Link |
| --- | --- | --- |
| ReVeal | IEEE TSE  | [Link](https://ieeexplore.ieee.org/abstract/document/9448435/?casa_token=S7Edzt0cuYkAAAAA:XId-rO6uAISCMYMyq4bvmcD83vqSfPCnZDqycv8iHI-tRZ9OVm-gAZzwIVZZGustUX1IsQ7Oew) |
| Devign | NeurIPS | [Link](https://proceedings.neurips.cc/paper_files/paper/2019/hash/49265d2447bc3bbfe9e76306ce40a31f-Abstract.html) |
| Draper | IEEE ICMLA |  [Link](https://arxiv.org/abs/1807.04320) |
| BigVul | IEEE ICSE | [Link](https://dl.acm.org/doi/abs/10.1145/3379597.3387501) |
| DiverseVul | IEEE ICSE  |  [Link](https://dl.acm.org/doi/abs/10.1145/3607199.3607242) |


## How to Run and Evaluate

We provided user-friendly shell scripts to simplify model training and evaluation. These scripts are located in the `scripts/` directory, and their general functionalities are as follows:

```shell
scripts
├── download_hf_models.py          # Download the Hugging Face models used in this repository into `model/`.
├── finetune.sh                     # Fine-tune LLMs for the experiments on Section 6.2 and 6.3.
├── inference.sh                    # Evaluate LLMs fine-tuned using `finetune.sh`.
├── finetune_imbalance.sh           # Fine-tune LLMs for the experiments on Section 6.4.
├── inference_imbalance.sh          # Evaluate LLMs fine-tuned using `finetune_imbalance.sh`.
├── finetune_ablation.sh            # Fine-tune LLMs for the experiments on Section 6.5.
├── inference_ablation.sh           # Evaluate LLMs fine-tuned using `finetune_ablation.sh`.
├── train.sh                        # Train graph-based and medium-size sequence models for the experiments on Section 6.2.
├── test.sh                         # Evaluate models trained using `train.sh`.
├── train_imbalance.sh              # Train graph-based and medium-size sequence models for the experiments on Section 6.4.
├── test_imbalance.sh               # Evaluate models trained using `train_imbalance.sh`.
└── to_graph.sh                     # Convert data into graph format as input for Devign model.
```

Before using these scripts, you need to:
1. Use the `cd` command to set the running directory to the root of this repository.
2. Place all data in the `data/` directory, following the directory structure and file names provided in our open-sourced HuggingFace repository.
3. Install all dependencies listed in the `requirements.txt` by running the command `pip install -r requirements.txt`.
4. Download the required Hugging Face base models if you plan to run the LLM-based experiments.

The trained models and output log will be generated in the `outputs/` directory.

If you want to download the Hugging Face models used by this repository into the `model/` directory, you can run:

```shell
python3 scripts/download_hf_models.py
```

You can also download only selected models:

```shell
python3 scripts/download_hf_models.py --models codebert graphcodebert unixcoder llama3.2
```

If you are in mainland China and want to use the community mirror, you can either pass the mirror shortcut:

```shell
python3 scripts/download_hf_models.py --mirror china --models codebert unixcoder llama3.2
```

or set a custom Hugging Face endpoint:

```shell
export HF_ENDPOINT=https://hf-mirror.com
python3 scripts/download_hf_models.py --models codebert unixcoder llama3.2
```

For gated repositories such as Llama / CodeLlama, make sure you have access permission on Hugging Face and set your token before downloading:

```shell
export HF_TOKEN=your_huggingface_token
python3 scripts/download_hf_models.py --mirror china --models llama3.2
```

### FastAPI backend for frontend integration

To let a frontend submit jobs and poll training or inference status, this repository now includes a FastAPI backend in `fastapi_backend/`.

Start the backend from the repository root:

```shell
pip install -r requirements.txt
./scripts/run_fastapi.sh 0.0.0.0 8000
```

After startup, you can open:

- `http://127.0.0.1:8000/docs` for Swagger UI
http://127.0.0.1:8000/docs

- `http://192.168.3.6:8000/health` for a health check

The main API groups are:

- `POST /api/frontend/code-inference`: run direct single-code inference with a trained CodeBERT or UniXcoder checkpoint
- `POST /api/jobs/classical`: trigger `train.sh` or `test.sh`
- `POST /api/jobs/classical-imbalance`: trigger `train_imbalance.sh` or `test_imbalance.sh`
- `POST /api/jobs/llm`: trigger `finetune.sh` or `inference.sh`
- `POST /api/jobs/llm-imbalance`: trigger `finetune_imbalance.sh` or `inference_imbalance.sh`
- `POST /api/jobs/ablation`: trigger `finetune_ablation.sh` or `inference_ablation.sh`
- `POST /api/jobs/to-graph`: trigger `to_graph.sh`
- `GET /api/jobs`: list all submitted jobs
- `GET /api/jobs/{job_id}`: query one job
- `GET /api/jobs/{job_id}/log`: fetch the tail of the log file for frontend polling

By default the backend enables CORS for all origins. You can restrict it with:

```shell
ALLOWED_ORIGINS=http://localhost:5173,http://127.0.0.1:3000 ./scripts/run_fastapi.sh
```

### Quick start

To quickly get started, you can run the following examples:

```shell
# For the experiments on Section 6.2.
./scripts/finetune.sh             reveal  llama3.1  0-512 16    0;
./scripts/inference.sh            reveal  llama3.1  0-512       0;
./scripts/train.sh                reveal  ReGVD     0-512       0;
./scripts/test.sh                 reveal  ReGVD     0-512       0;
# For the experiments on Section 6.3.
./scripts/finetune.sh             mix     llama3.1  128-256 32  0;
./scripts/inference.sh            mix     llama3.1  128-256     0;
./scripts/train.sh                mix     ReGVD     128-256     0;
./scripts/test.sh                 mix     ReGVD     128-256     0;
# For the experiments on Section 6.4.
./scripts/finetune_imbalance.sh   draper  llama3.1  0·2         0;
./scripts/inference_imbalance.sh  draper  llama3.1  0·2         0;
./scripts/train_imbalance.sh bigvul CodeBERT 0-512 1 0;
./scripts/train_imbalance_test.sh bigvul_cwe20 CodeBERT 1 1 0;
./scripts/test_imbalance.sh       draper  CodeBERT  0·2         0;
# For the experiments on Section 6.5.
./scripts/finetune_ablation.sh    reveal  llama3.1  8 16        0;
./scripts/inference_ablation.sh   reveal  llama3.1  8 16        0;
# Llama 3.2 is also supported in LLM scripts.
./scripts/finetune.sh             reveal  llama3.2  0-512 16    0;
./scripts/inference.sh            reveal  llama3.2  0-512       0;
./scripts/finetune_ablation.sh    bigvul  llama3.2  1 16        0;
./scripts/inference_ablation.sh   bigvul  llama3.2  1 16        0;
```

You can modify the command-line arguments in the above examples to perform other experiments mentioned in the paper.
For the ablation scripts, the arguments are `<DATASET_NAME> <MODEL_NAME> <R> <ALPHA> [CUDA]`.

### Rebucket alpaca data by token length

If you already have split Alpaca-format JSON files and want to rebucket them by token length in place, you can use:

```shell
venv/bin/python data_process/rebucket_alpaca_by_length.py \
  --dataset-dir data/cvefixes_cwe77 \
  --dataset-name cvefixes_cwe77 \
  --input-prefix cvefixes_cwe77_0-512
```

This script will:

- rename the original `alpaca/` directory to `alpaca1/`
- create a new empty `alpaca/` directory
- read `*_train.json`, `*_validate.json`, and `*_test.json` from `alpaca1/`
- compute token length from each sample's `input` field using `model/Llama-3.2-1B`
- write rebucketed files back into the new `alpaca/` directory
- save files with the naming rule `数据集名称_长度_train|validate|test.json`
- append a `token_length` field to each output sample
- delete `alpaca1/` after successful completion

For the example above, the generated files are placed under:

- `data/cvefixes_cwe77/alpaca/`

The output naming rule is:

- `<dataset_name>_0-512_train.json`
- `<dataset_name>_512-1024_train.json`
- `<dataset_name>_1024-*_train.json`
- the same pattern is used for `validate` and `test`

You can also customize the bucket boundaries or process only selected splits:

```shell
venv/bin/python data_process/rebucket_alpaca_by_length.py \
  --dataset-dir data/cvefixes_cwe77 \
  --dataset-name cvefixes_cwe77 \
  --input-prefix cvefixes_cwe77_0-512 \
  --bucket-boundaries 256 512 1024 \
  --splits train validate
```

### Reviewer fine-tuning workflow

If you want to fine-tune an LLM reviewer on top of the positive predictions made by a small model, you can use the following two-step workflow.

1. Generate reviewer CSV files from an existing small-model checkpoint directory under `outputs/<RESULT_MODEL>_imbalance/<DATASET>_<LENGTH>_<POS_RATIO>/`:

```shell
./scripts/test_imbalance_test.sh bigvul_cwe20 CodeBERT 1 1 0
```

This script reuses the existing checkpoint in `outputs/CodeBERT_imbalance/bigvul_cwe20_1_1/` and generates:

- `reviewer_train.csv`
- `reviewer_val.csv`
- `reviewer_test.csv`
- `results.csv` (compatibility copy of `reviewer_test.csv`)

2. Prepare reviewer JSON files and rebucket them by token length:

```shell
./data_process/rebucket_reviewer_data.sh bigvul_cwe20 CodeBERT 1 1 0
```

This script will:

- read `outputs/CodeBERT_imbalance/bigvul_cwe20_1_1/reviewer_train.csv`
- read `outputs/CodeBERT_imbalance/bigvul_cwe20_1_1/reviewer_val.csv`
- read `outputs/CodeBERT_imbalance/bigvul_cwe20_1_1/reviewer_test.csv`
- generate reviewer JSON files under `reviewer_finetune_data/`
- rebucket reviewer JSON files by token length

3. Start LoRA fine-tuning from an already prepared bucket:

```shell
./scripts/finetune_test.sh bigvul_cwe20 CodeBERT llama3.2 1 1 4 0
```

The arguments are:

- `<DATASET_NAME> <RESULT_MODEL_NAME> <LLM_MODEL_NAME> <LENGTH> <POS_RATIO> <BATCH_SIZE> [LENGTH_BUCKET] [CUDA]`

For the example above, the script will:

- read the already prepared rebucketed reviewer JSON files
- launch `LLM/finetuning_test.py` using the rebucketed train/val JSON files

The generated JSON files follow the naming rule:

- `reviewer_finetune_data/CodeBERT_imbalance/bigvul_cwe20_1_1/train.json`
- `reviewer_finetune_data/CodeBERT_imbalance/bigvul_cwe20_1_1/val.json`
- `reviewer_finetune_data/CodeBERT_imbalance/bigvul_cwe20_1_1/test.json`

After rebucketing, the script uses files under:

- `reviewer_finetune_data/CodeBERT_imbalance/bigvul_cwe20_1_1_length_rebucketed/`

For example:

- `train_0-512.json`
- `val_0-512.json`
- `test_0-512.json`

Each reviewer JSON sample keeps the following fields:

```json
{
  "instruction": "The small model predicts that the following code contains a vulnerability. Determine whether this prediction should be kept or rejected.",
  "input": "Small model confidence: 0.78\n\nCode:\n...",
  "prob": 0.78,
  "output": "0",
  "index": 103840
}
```

Notes:

- reviewer fine-tuning currently keeps only the samples with `Prediction = 1` from the small-model CSV
- `output = "1"` means `keep`
- `output = "0"` means `reject`
- `prob` is stored in the JSON and is also injected into the prompt as a standalone line before the code
- `finetuning_test.py` reads the prepared JSON files directly; it no longer needs to build training samples from CSV on the fly
- the default length bucket is `0-512`
- if you want to specify a bucket explicitly, for example `512-1024`, use:

```shell
./scripts/finetune_test.sh bigvul_cwe20 CodeBERT llama3.2 1 1 4 512-1024 0
```

### Reviewer fine-tuning with OOF small-model predictions

If you want the reviewer training set to use stricter out-of-fold (OOF) predictions from the small model, see the standalone guide:

- [`OOF_README.md`](OOF_README.md)

Short version:

1. Train the small model normally and export the original reviewer CSV files.
2. Run OOF to generate a new `reviewer_train.csv`.
3. Build reviewer JSON files with `OOF train + original val/test`.
4. Fine-tune the LoRA reviewer and evaluate it on the unchanged reviewer test split.

The detailed guide in [`OOF_README.md`](OOF_README.md) covers both imbalance and non-imbalance datasets.

Most commonly used scripts:

- [`scripts/train_imbalance_oof.sh`](scripts/train_imbalance_oof.sh)
- [`data_process/rebucket_reviewer_data_oof_train.sh`](data_process/rebucket_reviewer_data_oof_train.sh)
- [`scripts/finetune_test.sh`](scripts/finetune_test.sh)
- [`scripts/inference_finetune_test.sh`](scripts/inference_finetune_test.sh)

- if the 7th argument is a plain value like `0`, the script treats it as `CUDA`, and `LENGTH_BUCKET` falls back to `0-512`

Specifically, the second parameter represents the dataset name, which corresponds to the folder name in the `data/` directory.
You can customize a new dataset (assume it is named `xxx`) by following the template of our open-sourced dataset on the HuggingFace repository. Store it according to the following file structure:

```
data
└── xxx
    └── alpaca
        ├── xxx_0-123_test.json
        ├── xxx_0-123_train.json
        ├── xxx_0-123_validate.json
        ├── xxx_123-456_test.json
        ├── xxx_123-456_train.json
        ├── xxx_123-456_validate.json
        └── ...
```

The third parameter specifies the model name, which has the presetting supported values:

- For scripts prefixed with `finetune` and `inference`, the supported values are: `llama2`, `codellama`, `llama3`, `llama3.1`, and `llama3.2` (all lowercase).
- For scripts prefixed with `train` and `test`, the supported values are: `Devign`, `ReGVD`, `GraphCodeBERT`, `CodeBERT`, and `UniXcoder` (case-sensitive).

Other parameters can refer to the usage within every script.

### Run Devign model

Unlike other models, the Devign model requires data to be converted into graph format before training and evaluation. To simplify this process, we provide the `to_graph.sh` script.

Before converting, you need to download `joern.zip` from [this link](https://drive.google.com/file/d/1Jh5-JJ8BKGL53uAxWjSnEBjw0oRDm_Ss/view?usp=sharing), extract it, and store all the files in the `joern/` directory. Make sure the current user has execute (`x`) permissions for `joern-parse`.

An example of training and evaluating the Devign model is as follows:

```shell
./scripts/to_graph.sh   reveal  0-512;
./scripts/train.sh      reveal  Devign  0-512   0;
```


## Awesome Helpful Resources

We implement our studied models by referencing the following resources or codebases, and we also recommend some useful related resources for further study.

| Resource Name | Summary | Link |
| --- | --- | --- |
| VulLLM | Referenced Codebase for Implementation | [Link](https://github.com/CGCL-codes/VulLLM) |
| Devign | Referenced Codebase for Implementation | [Link](https://github.com/saikat107/Devign) |
| CodeBERT Family | Referenced Codebase for Implementation | [Link](https://github.com/microsoft/CodeBERT) |
| ReGVD | Referenced Codebase for Implementation | [Link](https://github.com/daiquocnguyen/GNN-ReGVD) |
| Llama Family | Meta AI Open-source LLMs | [Link](https://arxiv.org/abs/2307.09288) |
| Evaluate ChatGPT for CVD | Recommended Codebase | [Link](https://github.com/soarsmu/ChatGPT-VulDetection) |
| Awesome Code LLM | Recommended Paper List | [Link](https://github.com/PurCL/CodeLLMPaper) |
| Awesome LLM for Software Engineering | Recommended Paper List | [Link](https://github.com/gai4se/LLM4SE) |
| Awesome LLM for Security | Recommended Paper List | [Link](https://github.com/liu673/Awesome-LLM4Security) |
| Awesome Code Vulnerability Analysis| Recommended Paper List | [Link](https://github.com/huhusmang/Awesome-LLMs-for-Vulnerability-Detection) |


## Acknowledgement

We are very grateful that the authors of VulLLM, CodeLlama, Meta AI and other open-source efforts which make their codes or models publicly available so that we can carry out this experimental study on top of their hard works.


## Citing this work
If you find this codebase useful in your research, please consider citing our work and previous great works as follows.
By the way, collaboration and pull requests are always welcome! If you have any questions or suggestions, please feel free to contact us : )

```bibtex
@article{jiang2024investigating,
  title={Investigating Large Language Models for Code Vulnerability Detection: An Experimental Study},
  author={Jiang, Xuefeng and Wu, Lvhua and Sun, Sheng and Li, Jia and Xue, Jingjing and Wang, Yuwei and Wu, Tingting and Liu, Min},
  journal={arXiv preprint},
  year={2024}
}

@article{feng2020codebert,
  title={Codebert: A pre-trained model for programming and natural languages},
  author={Feng, Zhangyin and Guo, Daya and Tang, Duyu and Duan, Nan and Feng, Xiaocheng and Gong, Ming and Shou, Linjun and Qin, Bing and Liu, Ting and Jiang, Daxin and others},
  journal={arXiv preprint arXiv:2002.08155},
  year={2020}
}

@article{du2024generalization,
  title={Generalization-Enhanced Code Vulnerability Detection via Multi-Task Instruction Fine-Tuning},
  author={Du, Xiaohu and Wen, Ming and Zhu, Jiahao and Xie, Zifan and Ji, Bin and Liu, Huijun and Shi, Xuanhua and Jin, Hai},
  journal={arXiv preprint arXiv:2406.03718},
  year={2024}
}
```
